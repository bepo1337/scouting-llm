# LLM4Scouting
General explanation abt the proj

## Disclaimer 
This project is a university project for the course [Web Interfaces for Language Processing Systems](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/ma-projects/master-project-web-interfaces.html) at the [University of Hamburg](https://www.uni-hamburg.de/).
Explaine xperiments folder
# Install 
`pip install -r requirements.txt`

# Configuration
- need .env file with credentials for OpenAI
- Alternatively: Ollama running (TODO need to implement different docker images? or a starting parameter?)
- Activate environment with requirements installed 
- 
# Initialize data loading into vector and relational database
`docker compose up -d` to start PostgreSQL (localhost:5432) and Milvus (localhost:19530)
# Run
`flask run` or `make run`

# Features
### Scouting as Prompt
Explain each field and give example prompts. Also explain that we will only find something with the query if its within the scouting reports. So ie if we have data from the MSL, its not very likely we will find something with the prompt "im looking for a defender that can play well in the third german league and has a proven record to be a leader there". Also reactions that they re stored and could be mined later, but are of no use in this app 
### Comparing Players
### Visualizing Players in a Network
The feature allows you to compare soccer players based on their playstyle. You start by searching for a player in the database using a search bar. The selected player is then depicted at the center of a dynamic network graph, surrounded by other players who share similar playstyles. These similarities are determined by the vector representation of each player's report summary.

In this network, the thickness of the edges between players reflects the degree of similarity in their playstyles—thicker edges indicate stronger similarity, while thinner edges indicate less similarity. You can interact with the network by double-clicking on any player to expand the view, revealing additional players with similar playstyles, thus growing the network. This allows for an in-depth exploration of connections between multiple players.

If you're interested in a specific player, a single click on that player's node will open a detailed summary of their reports. You can also click on the edge between two players to directly compare them, offering insights into their playstyle similarities and differences.

# Build and run docker image
`docker build --tag scouting-llm .`
`docker run -p 5000:5000 scouting-llm`

## Authors

* **Benjamin Pöhlmann** - [bepo1337](https://github.com/bepo1337)

* **Marvin Schmohl**    - [marvinschmohl](https://github.com/marvinschmohl)
