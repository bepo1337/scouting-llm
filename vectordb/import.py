import argparse
import json
from dataclasses import dataclass

import numpy as np
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
parser.add_argument("--collection", default="scouting", nargs="?",
                    help="Name of collection to insert (default: scouting)")
parser.add_argument("--milvus-port", default="19530", nargs="?",
                    help="Port Milvus is running on (default: 19530)")
args, unknown = parser.parse_known_args()

import_file = args.file
milvus_port = args.milvus_port
collection_name = args.collection

dimensions = 8


@dataclass
class Report:
    scout_id: str
    text: str
    player_id: str
    player_transfermarkt_id: str
    grade_rating: float
    grade_potential: float


def create_collection() -> Collection:
    print(f"... Creating collection {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="scout_id", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="player_id", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="player_transfermarkt_id", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="grade_rating", dtype=DataType.FLOAT),
        FieldSchema(name="grade_potential", dtype=DataType.FLOAT),
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


def json_to_reports() ->[Report]:
    with open(import_file) as f:
        data = json.load(f)
        reports = [Report(**item) for item in data]

    return reports

def create_embedding(text: str) -> DataType.FLOAT_VECTOR:
    # for now just random vector, have to see how to embed with model
    rng = np.random.default_rng(seed=19530)
    return rng.random(dimensions)

def create_embeddings(reports: [Report]) -> [DataType.FLOAT_VECTOR]:
    return [create_embedding(item.text) for item in reports]


def import_reports(collection: Collection, reports: [Report]):
    print(f"... Inserting reports from {import_file}")
    # Have to work with append or smth for large files cuz we iterate over whole report set a lot of times
    reports_in_batch_insert_format = [
        [item.scout_id for item in reports],
        [item.text for item in reports],
        [item.player_id for item in reports],
        [item.player_transfermarkt_id for item in reports],
        [item.grade_rating for item in reports],
        [item.grade_potential for item in reports],
        create_embeddings(reports)
    ]

    insert_result = collection.insert(reports_in_batch_insert_format)
    collection.flush()
    print(f"Insertion result: Successes: {insert_result.succ_count}, Errors: : {insert_result.err_count}")
    print(f"Total number of reports in Milvus: {collection.num_entities}")

# Setup Milvus connection
collection = setup_milvus_connection()
print("... Connected to Milvus, collection: {}".format(collection.name))

# Read JSON file and cast to our Report class
reports = json_to_reports()

# Import reports to collection
import_reports(collection, reports)

# Can now query results with milvus_cli https://milvus.io/docs/cli_overview.md
# or query_example.py
