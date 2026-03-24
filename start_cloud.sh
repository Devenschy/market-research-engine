#!/bin/bash
# start_cloud.sh — Railway startup script
# Runs both the trading engine and the Streamlit dashboard together

# Create logs directory if it doesn't exist
mkdir -p logs

# Download NLTK data needed by sentiment.py
python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Start the trading engine in the background
echo "[startup] Starting trading engine..."
python main.py &

# Start Streamlit in the foreground (Railway needs this process to stay alive)
echo "[startup] Starting Streamlit dashboard..."
streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
