import argparse
import copy
import json
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import TokenTextSplitter

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    DataType,
    Collection,
    CollectionSchema
)

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="reports.json", nargs="?",
                    help="What file to import (default: reports.json)")
parser.add_argument("--collection", default="test_scouting", nargs="?",
                    help="Name of collection to insert (default: scouting_summary)")
parser.add_argument("--milvus-port", default="19530", nargs="?",
                    help="Port Milvus is running on (default: 19530)")
parser.add_argument("--chunk-size", type=int, default=70, nargs="?",
                    help="Size of the chunks that we embed (default: 70)")
parser.add_argument("--chunk-overlap", type=int, default=15, nargs="?",
                    help="Size of the chunk overlap (default: 15)")

args, unknown = parser.parse_known_args()

import_file = "data/" + args.file
milvus_port = args.milvus_port
collection_name = args.collection
chunk_size = args.chunk_size
chunk_overlap = args.chunk_overlap

# Have to change together. Cant set dimensions per model withput PCA or other dimensionality reduction method.
dimensions = 768
embedding_model = "nomic-embed-text"

# Want json file with Report for all player ids

# Create a map player_id -> [Report]
# For each key in map:
    # Get summary from llm
    # Average grade rating
    # Average grade potential
    # use first seen main_position/played_position as played_position
    # ignore scoutid

@dataclass
class Report:
    scout_id: str
    text: str
    player_id: str
    player_transfermarkt_id: str
    grade_rating: Optional[float]
    grade_potential: Optional[float]
    main_position: str
    played_position: str


def create_collection() -> Collection:
    print(f"... Creating collection {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="player_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="player_transfermarkt_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="scout_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="grade_rating", dtype=DataType.FLOAT),
        FieldSchema(name="grade_potential", dtype=DataType.FLOAT),
        FieldSchema(name="main_position", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="played_position", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dimensions)
    ]

    schema = CollectionSchema(fields, "scouting report collection")
    scouting_collection = Collection(collection_name, schema, consistency_level="Strong")
    return scouting_collection


def setup_milvus_connection() -> Collection:
    connections.connect("default", host="localhost", port=milvus_port)
    collection_exists = utility.has_collection(collection_name)
    if not collection_exists:
        return create_collection()

    print(f"... Collection {collection_name} already exists")
    return Collection(collection_name)


def json_to_reports() -> [Report]:
    with open(import_file) as f:
        data = json.load(f)
        reports = [Report(
            scout_id=item.get('scout_id'),
            text=item.get('text'),
            player_id=item.get('player_id'),
            player_transfermarkt_id=item.get('player_transfermarkt_id'),
            grade_rating=item.get('grade_rating') if item.get('grade_rating') is not None else 0.0,
            grade_potential=item.get('grade_potential') if item.get('grade_potential') is not None else 0.0,
            main_position=item.get('main_position'),
            played_position=item.get('played_position')
        ) for item in data]

    return reports


def copy_report_with_new_chunk(report: Report, chunk: str) -> Report:
    copied_report = copy.deepcopy(report)
    copied_report.text = chunk
    return copied_report


def chunk_reports(reports: [Report]) -> [Report]:
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_reports = []
    for report in reports:
        splitted_texts = splitter.split_text(report.text)

        for chunk in splitted_texts:
            chunked_report = copy_report_with_new_chunk(report, chunk)
            chunked_reports.append(chunked_report)

    return chunked_reports


def create_embeddings(reports: [Report]) -> [DataType.FLOAT_VECTOR]:
    report_texts = [item.text for item in reports]
    embeddings = OllamaEmbeddings(model=embedding_model)

    embedded_report_texts = []
    for text in tqdm(report_texts, desc="Embedding reports"):
        embedded_report_texts.append(embeddings.embed_documents([text])[0])

    return embedded_report_texts


def import_reports(collection: Collection, reports: [Report]):
    print(f"... Inserting reports from {import_file}")
    texts = []
    player_ids = []
    player_transfermarkt_ids = []
    scout_ids = []
    grade_ratings = []
    grade_potentials = []
    main_positions = []
    played_positions = []
    embeddings = create_embeddings(reports)

    # Iterate over reports once and populate the lists
    for report in reports:
        texts.append(report.text)
        player_ids.append(report.player_id)
        player_transfermarkt_ids.append(report.player_transfermarkt_id)
        scout_ids.append(report.scout_id)
        grade_ratings.append(report.grade_rating)
        grade_potentials.append(report.grade_potential)
        main_positions.append(report.main_position)
        played_positions.append(report.played_position)

    # Combine lists into the desired format
    reports_in_batch_insert_format = [
        texts,
        player_ids,
        player_transfermarkt_ids,
        scout_ids,
        grade_ratings,
        grade_potentials,
        main_positions,
        played_positions,
        embeddings
    ]

    insert_result = collection.insert(reports_in_batch_insert_format)
    collection.flush()
    print(f"Insertion result: Successes: {insert_result.succ_count}, Errors: : {insert_result.err_count}")
    print(f"Total number of reports in Milvus: {collection.num_entities}")


def create_embeddings_index(collection: Collection):
    print(f"... Creating index IVF_FLAT, L2, nlist=128 on 'embeddings'")
    index = {
        "index_type": "IVF_FLAT",
        # we cluster our data and only compare our query to the elements of the nearest cluster center https://milvus.io/docs/index.md#IVFFLAT
        "metric_type": "L2",  # euclidean distance, could also use cosine here https://milvus.io/docs/metric.md
        "params": {"nlist": 128},  # nlist -> number of clusters
    }

    collection.create_index("embeddings", index)


# Setup Milvus connection
collection = setup_milvus_connection()
print("... Connected to Milvus, collection: {}".format(collection.name))

# Read JSON file and cast to our Report class
reports = json_to_reports()

# Chunk documents
chunked_reports = chunk_reports(reports)

# Import reports to collection
import_reports(collection, chunked_reports)

# Create index on embeddings
create_embeddings_index(collection)

# Can now query results with milvus_cli https://milvus.io/docs/cli_overview.md
# or query_example.py
