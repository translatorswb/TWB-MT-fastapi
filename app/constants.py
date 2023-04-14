MOSES_TOKENIZER_DEFAULT_LANG = 'en'
HELSINKI_NLP = 'Helsinki-NLP'
MULTIMODALCODE = 'MULTI'
SUPPORTED_MODEL_TYPES = ['opus', 'opus-big', 'ctranslator2', 'dummy', 'custom', 'nllb']
MODEL_TAG_SEPARATOR = '-'

#This dictionary is needed to map language codes specified in config to the codes used in NLLB models. 
#Complete list of languages can be found in https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
#You can also use the NLLB id directly in the configuration file
NLLB_LANGS_DICT = {'en': 'eng_Latn', 'tr': 'tur_Latn', 'fr': 'fra_Latn',
				   'kr': 'knc_Latn', 'ha': 'hau_Latn', 'ff': 'fuv_Latn',
				   'rw': 'kin_Latn'}

NLLB_CHECKPOINT_IDS = ["nllb-200-distilled-1.3B", "nllb-200-distilled-600M", "nllb-200-3.3B"]