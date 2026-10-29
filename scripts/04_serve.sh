#!/usr/bin/env bash
# Independent services + UI.  Retrieval and generation scale separately.
cd "$(dirname "$0")/.."
case "${1:-all}" in
  retrieval)  uvicorn agri.retrieval.api:app --port 8001 ;;
  generation) uvicorn agri.generation.api:app --port 8002 ;;
  api)        uvicorn agri.service:app --port 8000 ;;
  app)        streamlit run app/streamlit_app.py ;;
  *) echo "usage: $0 retrieval|generation|api|app   (set retrieval.mode / generation.mode to remote in configs/config.yaml to use the services)";;
esac
