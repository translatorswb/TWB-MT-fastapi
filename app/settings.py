import os

CONFIG_JSON_PATH: str = os.getenv('MT_API_CONFIG', '')
MODELS_ROOT_DIR: str = os.getenv('MODELS_ROOT', '')
CTRANSLATE_DEVICE: str = (
    'cuda' if os.getenv('MT_API_DEVICE') == 'gpu' else 'cpu'
)
TRANSFORMERS_DEVICE: str = (
    0 if os.getenv('MT_API_DEVICE') == 'gpu' else -1)
CTRANSLATE_INTER_THREADS: int = int(os.getenv('MT_API_THREADS', 0)) or 16

#Specify which NLLB model to load here by default (if not specified in config as nllb_checkpoint_id)
DEFAULT_NLLB_MODEL_TYPE = "nllb-200-distilled-600M" # OR "nllb-200-distilled-1.3B" #"nllb-200-distilled-600M" #"nllb-200-3.3B" #facebook/nllb-200-1.3B
