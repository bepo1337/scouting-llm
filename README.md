# LLM4Scouting
In this project we implemented a Retrieval Augmented Generation (RAG) application for the soccer domain. Soccer clubs spend a lot of money by transferring new players to their team. For this the professional teams need dedicated departments for scouting which in turn produce a lot of reports that contain qualitative data.  \
With this we tried to reduce the necessity to go through a lot of scouting reports for single players but provide a scout with the tool to send a prompt to our application which will return suitable players from the database that match the prompt. \
This could greatly reduce the time to find new players and increase scout productivity. \
Additionally we implemented ways to compare players and also to visualize how similar players are to each other in a graph.

## Disclaimer 
This project is a university project for the course [Web Interfaces for Language Processing Systems](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/ma-projects/master-project-web-interfaces.html) at the [University of Hamburg](https://www.uni-hamburg.de/). \
The `/experiments` folder contains previous iterations and small experiments we did with different technologies. Those should not be held to a high standard as they're just prototypes.
# Install 
To install this application, you need Python 3.10+ installed and a virtual environment setup or just install the dependencies globally with: \
`pip install -r requirements.txt`

# Configuration
- `.env` file that contains the following:
  - `AZURE_OPENAI_API_KEY`
  - `OPENAI_API_VERSION`
  - `AZURE_OPENAI_ENDPOINT`
- Activate environment with requirements installed 

# Preparing and inserting data
## Convert data into a suitable format
In order to be imported, the original scouting reports need have the format of the reports in a JSON list under `data/reports.json`.
# Initialize data by loading reports into vector and relational database
`docker compose up -d` to start PostgreSQL (localhost:5432) and Milvus (localhost:19530) \
Alternatively run the whole stack already and then restart it again later after the data has been imported: \
`docker compose -f compose-stack.yml up` \
Once Milvus and Postgres are up, you can run the following commands to insert your original reports: \
`python3 import.py --file=<PATH_TO_YOUR_ORIGINAL_REPORTS> --collection=original_reports` which will create a collection with the name "original_reports". \

Once the reports have been imported, we can summarize them by running the following command: \
`python3 create_summary_reports.py --import=<PATH_TO_YOUR_ORIGINAL_REPORTS> --output=<PATH_TO_SUMMARY_REPORTS>` \

After the reports have been summarized successfully, you can run the following command to import them into Milvus:
`python3 import.py --file=<PATH_TO_YOUR_SUMMARIZED_REPORTS> --collection=summary_reports` which will create a collection with the name "summary_reports".

Finally we can import the data into our relational database aswell by running: \
`python3 import_rdb.py`
You will find examples for how the data is supposed to be structured as JSON in the `/data` folder.
# Run
`flask run` or `make run`

# Features
### Scouting as Prompt
Scouting as a Prompt is giving scouts the possibility to find players that are a match to the prompt they give. \
The given prompt will be expanded by an LLM to match the structure of the summaries which we save in the Milvus collection. Through this we make sure that the result is less dependent on the style of the prompt. 

The switch `fine grained` will search for the embedded prompt in the original reports and then return the players that match the prompt without any enhancement by an LLM. Through this we can not only search in summaries but also find attributions that may be rare to find in summaries. \
The filter `position` will only look for players that have this position as their `main_position` in the original reports. 

By using this approach, the application will only find players that are described similarly in the scouting reports. Therefore the usefulness depends on the size of the data set and the quality and thoroughness of the reports. 
While players that are scarcely described will have very superficial summary, summaries of players with a lot of detailed reports can be nuanced. 
Synonymous concepts will be found with the same prompt but if a player with a certain attribute doesn't exist in the database, no found player might be suitable.

Example prompts:
While `striker that can rock the EFL League Two` might not yield good results as "EFL League Two" might not be described in any report, a more detailed query such as `good query` might, as it gives more information to be embedded and match players from the database. \

There are reaction buttons under each player. The results are logged and have no fixed meaning. They could be interpreted as answers to questions such as "sign player?" but also as "good summary?". \
Explain each field and give example prompts. Also explain that we will only find something with the query if its within the scouting reports. So ie if we have data from the MSL, its not very likely we will find something with the prompt "im looking for a defender that can play well in the third german league and has a proven record to be a leader there". Also reactions that they re stored and could be mined later, but are of no use in this app

Reactions
### Comparing Players
### Visualizing Players in a Network
The feature allows you to compare soccer players based on their playstyle. You start by searching for a player in the database using a search bar. The selected player is then depicted at the center of a dynamic network graph, surrounded by other players who share similar playstyles. These similarities are determined by the vector representation of each player's report summary.

In this network, the thickness of the edges between players reflects the degree of similarity in their playstyles—thicker edges indicate stronger similarity, while thinner edges indicate less similarity. You can interact with the network by double-clicking on any player to expand the view, revealing additional players with similar playstyles, thus growing the network. This allows for an in-depth exploration of connections between multiple players.

If you're interested in a specific player, a single click on that player's node will open a detailed summary of their reports. You can also click on the edge between two players to directly compare them, offering insights into their playstyle similarities and differences.

# Build and run docker image locally

## Build and run just the backend
`docker build --tag scouting-llm-backend:1.0.0 .` \
`docker run -p 5000:5000 scouting-llm-backend:1.0.0`

# Build and run the full stack on a single host
Build [frontend](https://github.com/bepo1337/scouting-llm-frontend) and backend with version 1.0.0 each. \
Then run:
`docker compose -f compose-stack.yml up`

## Authors

* **Benjamin Pöhlmann** - [bepo1337](https://github.com/bepo1337)

* **Marvin Schmohl**    - [marvinschmohl](https://github.com/marvinschmohl)
