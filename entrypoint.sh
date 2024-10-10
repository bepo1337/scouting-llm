#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve and run nomic-embed-text model..."
ollama run nomic-embed-text
echo "🟢 Done!"

echo "🔴 Retrieve and run latest Llama..."
ollama run llama3:latest
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid