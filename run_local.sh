export MT_API_CONFIG=config.json
export MODELS_ROOT=../translation-models
uvicorn main:app --reload --port 8001
