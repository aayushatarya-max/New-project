#!/bin/bash

# Start FastAPI backend server in background on internal port 8000
echo "Starting FastAPI Backend on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Set API_URL for Streamlit frontend
export API_URL="http://localhost:8000/api"

# Wait 2 seconds for backend initialization
sleep 2

# Start Streamlit UI on Render's public PORT (defaults to 10000)
PORT="${PORT:-10000}"
echo "Starting Streamlit UI on port $PORT..."
exec streamlit run frontend/app.py --server.port "$PORT" --server.address 0.0.0.0
