import os

CONFIG_JSON_PATH: str = os.getenv('MT_API_CONFIG', '')
MODELS_ROOT_DIR: str = os.getenv('MODELS_ROOT', '')
CTRANSLATE_DEVICE: str = (
    'cuda' if os.getenv('MT_API_DEVICE') == 'gpu' else 'cpu'
)
TRANSFORMERS_DEVICE: str = (
    0 if os.getenv('MT_API_DEVICE') == 'gpu' else -1)
CTRANSLATE_INTER_THREADS: int = int(os.getenv('MT_API_THREADS', 0)) or 16
