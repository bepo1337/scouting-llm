# scouting-llm

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
### Visualizing Players in Graph
Marvin TODO

# Build and run docker image
`docker build --tag scouting-llm .`
`docker run -p 5000:5000 scouting-llm`
