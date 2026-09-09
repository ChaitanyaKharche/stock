#!/bin/bash
# Space entrypoint: FastAPI on :7860 (internal) + Streamlit on :8501 (the exposed port).
#
# The old version was two lines -- background uvicorn with `&`, then start Streamlit -- and
# never checked whether the API came up. That matters because config.py raises ImportError
# when FINNHUB_KEY or REDDIT_CLIENT_ID is unset, which kills uvicorn instantly. Streamlit
# then starts fine, the Space reports "Running" in green, and every click returns
# "Failed to connect to the API" with no indication that the actual cause is a missing
# secret. Same failure shape as every other bug in this codebase: a real error absorbed by
# the layer above and re-presented as something vague.
#
# So: wait for the API to actually answer, and if it does not, write the reason somewhere
# the UI can read and display. Streamlit still starts either way -- a Space that boots and
# explains itself is far more useful than one that exits.
set -u

STATUS_FILE=/tmp/api_status
API_LOG=/tmp/api.log
: > "$STATUS_FILE"

echo "[run.sh] starting FastAPI on :7860 ..."
uvicorn trade_analysis.enhanced_api:app --host 0.0.0.0 --port 7860 > "$API_LOG" 2>&1 &
API_PID=$!

# python, not curl -- python:3.12-slim ships no curl and adding it just for a health check
# is a layer nobody needs.
ready=0
for _ in $(seq 1 45); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        {
            echo "The analysis API exited while starting up."
            echo
            echo "Most common cause: a required Space secret is not set."
            echo "Settings -> Variables and secrets: FINNHUB_KEY, REDDIT_CLIENT_ID,"
            echo "REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT."
            echo
            echo "--- last lines of the API log ---"
            tail -n 25 "$API_LOG"
        } > "$STATUS_FILE"
        echo "[run.sh] API process died during startup; see $STATUS_FILE"
        break
    fi
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=2)" 2>/dev/null; then
        echo "ready" > "$STATUS_FILE"
        ready=1
        echo "[run.sh] API is answering."
        break
    fi
    sleep 2
done

if [ "$ready" -eq 0 ] && [ ! -s "$STATUS_FILE" ]; then
    {
        echo "The analysis API did not answer within 90 seconds of starting."
        echo "It may still be loading the sentiment models on first boot -- reload shortly."
        echo
        echo "--- last lines of the API log ---"
        tail -n 25 "$API_LOG"
    } > "$STATUS_FILE"
    echo "[run.sh] API health check timed out; starting the UI anyway."
fi

echo "[run.sh] starting Streamlit on :8501 ..."
exec streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
