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
The input data for this application should be in JSON format, with each scouting report represented as an individual object within a JSON array. In order to be imported, the original scouting reports need have the format of the reports in a JSON list under `data/reports.json`. \
This file can also be used to go through the whole import and summarizing process.

Each scouting report object should follow this structure:


```jsonc
{
  "scout_id": "1234",                       // Unique identifier for the scout
  "text": "Text of the scouting report",    // Detailed qualitative data about the player    
  "player_id": "5f99f90d13f011e59dfc42d3",  // Unique identifier for the player in the system
  "player_transfermarkt_id": "238716",      // Player's identifier from Transfermarkt or other reference platforms
  "grade_rating": 0.3,                      // Current performance rating (e.g., 0.0 to 1.0 scale)
  "grade_potential": 0.2,                   // Potential for future performance (e.g., 0.0 to 1.0 scale)
  "main_position": "centerforward",         // Player's primary position
  "played_position": "centerforward"        // Position played in the reported match
}
```
Only `text`, `player_transfermarkt_id` and `main_position` are required. The others can be left with placeholders. 

All import file names will be prefixed with `data/` so only specify the path relative to the `data` directory.
# Initialize data by loading reports into vector and relational database
`docker compose up -d` to start PostgreSQL (localhost:5432) and Milvus (localhost:19530) \
Alternatively run the whole stack already and then restart it again later after the data has been imported: \
`docker compose -f compose-stack.yml up` 

Once Milvus and Postgres are up, you can run the following commands to insert your original reports: \
`python3 import.py --file=<PATH_TO_YOUR_ORIGINAL_REPORTS> --collection=original_reports` \
which will create a collection with the name "original_reports". 

Once the original reports have been imported, we can summarize them by running the following command: \
`python3 create_summary_reports.py --importfile=<PATH_TO_YOUR_ORIGINAL_REPORTS> --outputfile=<PATH_TO_SUMMARY_REPORTS>` 

After the reports have been summarized successfully, you can run the following command to import them into Milvus: \
`python3 import.py --file=<PATH_TO_SUMMARY_REPORTS> --collection=summary_reports` which will create a collection with the name "summary_reports".

Finally we can import the data into our relational database aswell by running: \
`python3 import_rdb.py --file=<PATH_TO_YOUR_ORIGINAL_REPORTS>` \
You will find examples for how the data is supposed to be structured as JSON in the `data/reports.json` file.

# Run locally
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

There are reaction buttons under each player. The results are logged and have no fixed meaning. They could be interpreted as answers to questions such as "sign player?" but also as "good summary?". 

### Comparing Players
This features creates custom comparisons of two players. A comparison will always include a general and a conclusion comparison. 
A LLM will receive the summaries of the players and the original reports and is instructed with a prompt to create a comparison. \
The comparison can be customized by either turning on the provided switches, which are more general: 
- Offensive Capabilities
- Defensive Capabilities
- Strenghts
- Weaknesses

Or there is also the possibility to create custom comparison topics. They can be input in the prompt area and have to be separated by a semicolon ";". This enables comparison for detailed attributes that might not be generally applicable such as "leadership skills" or "passing in the last third".


If a comparison topic is not applicable to the players, it will be left out in the response.
### Visualizing Players in a Network
The feature allows you to compare soccer players based on their playstyle. You start by searching for a player in the database using a search bar. The selected player is then depicted at the center of a dynamic network graph, surrounded by other players who share similar playstyles. These similarities are determined by the vector representation of each player's report summary.

In this network, the thickness of the edges between players reflects the degree of similarity in their playstyles—thicker edges indicate stronger similarity, while thinner edges indicate less similarity. You can interact with the network by double-clicking on any player to expand the view, revealing additional players with similar playstyles, thus growing the network. This allows for an in-depth exploration of connections between multiple players.

If you're interested in a specific player, a single click on that player's node will open a detailed summary of their reports. You can also click on the edge between two players to directly compare them, offering insights into their playstyle similarities and differences.

# Build and run docker image locally

## Build and run just the backend
Build: \
`docker build --tag scouting-llm-backend:1.0.0 .` \
Run: \
`docker run -p 5000:5000 scouting-llm-backend:1.0.0`

# Build and run the full stack on a single host
Build [frontend](https://github.com/bepo1337/scouting-llm-frontend) and backend with version 1.0.0 each (or use other versions and change the `compose-stack.yml`). \
Then run:
`docker compose -f compose-stack.yml up`

## Authors

* **Benjamin Pöhlmann** - [bepo1337](https://github.com/bepo1337)

* **Marvin Schmohl**    - [marvinschmohl](https://github.com/marvinschmohl)
