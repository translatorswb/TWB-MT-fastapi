import os

CONFIG_JSON_PATH = os.getenv('MT_API_CONFIG')
MODELS_ROOT_DIR = os.getenv('MODELS_ROOT')
CTRANSLATE_DEVICE = 'cuda' if os.getenv('MT_API_DEVICE')=='gpu' else 'cpu'
CTRANSLATE_INTER_THREADS = int(os.getenv('MT_API_THREADS')) or 16
MOSES_TOKENIZER_DEFAULT_LANG = 'en'
HELSINKI_NLP = 'Helsinki-NLP'
SUPPORTED_MODEL_TYPES = ['opus', 'ctranslator2', 'dummy']
MODEL_TAG_SEPARATOR = '-'
