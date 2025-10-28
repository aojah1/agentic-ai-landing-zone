# Agent Orchestrator

## Run
python3.13 -m venv .venv_client_app

source .venv_client_app/bin/activate   

python3.13 -m pip install -r requirements.txt  

python3.13 -m uvicorn src.app.orderxhub.fastapi_orderx:app --reload --host 0.0.0.0 --port 8084

streamlit run app.py --server.address 0.0.0.0 --server.port 8505   