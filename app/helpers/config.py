import json
import logging
import os

from app.constants import (
    CONFIG_JSON_PATH,
    MODELS_ROOT_DIR,
    SUPPORTED_MODEL_TYPES,
)
from app.helpers.singleton import Singleton
from app.utils.segmenters import (
    desegmenter,
    get_bpe_segmenter,
    get_sentencepiece_desegmenter,
    get_sentencepiece_segmenter,
    nltk_sentence_segmenter,
    token_desegmenter,
    token_segmenter,
)
from app.utils.tokenizers import (
    get_custom_tokenizer,
    get_moses_detokenizer,
    get_moses_tokenizer,
)
from app.utils.translators import (
    get_batch_ctranslator,
    get_batch_opustranslator,
    dummy_translator,
)
from app.utils.utils import (
    capitalizer,
    get_model_id,
    lowercaser,
    parse_model_id,
)

logger = logging.getLogger('console_logger')


class ConfigurationException(Exception):
    pass


class Config(metaclass=Singleton):
    def __init__(self):
        self.loaded_models = {}
        self.config_data = {}
        self.language_codes = {}
        self.languages_list = {}

        self._validate()
        self._load_models()
        self._load_languages_list()

    def map_lang_to_closest(self, lang):
        if lang in self.language_codes:
            return lang
        elif '_' in lang:
            superlang = lang.split('_')[0]
            if superlang in self.language_codes:
                return superlang
        return ''

    def _load_languages_list(self):
        for model_id in self.loaded_models.keys():
            source, target, alt = parse_model_id(model_id)
            if not source in self.languages_list:
                self.languages_list[source] = {}
            if not target in self.languages_list[source]:
                self.languages_list[source][target] = []

            self.languages_list[source][target].append(model_id)

    def _validate_config_file(self):
        if not os.path.exists(CONFIG_JSON_PATH):
            logger.warning(
                f'Config file {CONFIG_JSON_PATH} not found. '
                'No models will be loaded.'
            )
        else:
            try:
                with open(CONFIG_JSON_PATH, 'r') as jsonfile:
                    self.config_data = json.load(jsonfile)
            except json.decoder.JSONDecodeError:
                msg = 'Config file format broken. No models will be loaded.'
                logger.error(msg)
                raise ConfigurationException(msg)

    def _validate_models(self):
        if not os.path.exists(MODELS_ROOT_DIR):
            msg = '`models` directory not found. No models will be loaded.'
            logger.error(msg)
            raise ConfigurationException(msg)

        if not 'models' in self.config_data:
            msg = "Model spefication list ('models') not found in configuration."
            logger.error(msg)
            raise ConfigurationException(msg)

    def _validate_languages(self):
        if 'languages' in self.config_data:
            self.language_codes = self.config_data['languages']
            logger.debug(f'Languages: {self.language_codes}')
        else:
            logger.warning(
                "Language name spefication dictionary ('languages') not found in configuration."
            )

    def _validate(self):
        # Check if config file is there and well formatted
        self._validate_config_file()

        # Check if MODELS_ROOT_DIR exists
        self._validate_models()

        self._validate_languages()

    def _is_valid_model_config(self, model_config):
        # Check if model_type src and tgt fields are specified
        for item in ['src', 'tgt', 'model_type']:
            if item not in model_config:
                logger.warning(
                    f'`{item}` not speficied for a model. Skipping load'
                )
                return False
        return True

    def _is_valid_model_type(self, model_type):
        if not model_type in SUPPORTED_MODEL_TYPES:
            logger.warning(
                f'`model_type` not recognized: {model_type}. Skipping load'
            )
            return False
        return True

    def _validate_src_tgt(self, src, tgt):
        if not src in self.language_codes:
            logger.warning(
                f'Source language code `{src}` not defined in '
                'languages dict. This will surely break something.'
            )
        if not tgt in self.language_codes:
            logger.warning(
                f'Target language code `{tgt}` not defined in '
                'languages dict. This will surely break something.'
            )

    def _get_model_path(self, model_config, model_id):
        model_dir = None

        # Check model path
        if 'model_path' in model_config and model_config['model_path']:
            model_dir = os.path.join(
                MODELS_ROOT_DIR, model_config['model_path']
            )
            if not os.path.exists(model_dir):
                model_dir = None
                logger.warning(
                    f'Model path {model_dir} not found for model {model_id}. '
                    "Can't load custom translation model or segmenters."
                )
        else:
            logger.warning(
                f'Model path not specified for model {model_id}. '
                "Can't load custom translation model or segmenters."
            )
        return model_dir

    def _load_models(self):
        for model_config in self.config_data['models']:
            if not 'load' in model_config or not model_config['load']:
                continue

            # CONFIG CHECKS
            if not self._is_valid_model_config(model_config):
                continue

            if not self._is_valid_model_type(model_config['model_type']):
                continue

            # Load model variables
            model = {}
            model['src'] = model_config['src']
            model['tgt'] = model_config['tgt']
            alt_id = model_config.get('alt')

            model_id = get_model_id(
                model_config['src'], model_config['tgt'], alt_id
            )

            # Check if language names exist for the language ids
            self._validate_src_tgt(model['src'], model['tgt'])

            model_dir = self._get_model_path(model_config, model_id)

            # More configuration checks
            # Check conflicting subword segmenters
            if (
                'bpe' in model_config['pipeline']
                and 'sentencepiece' in model_config['pipeline']
            ):
                logger.warning(
                    f'Model {model_id} has both sentencepiece and bpe setup. '
                )

            # Check conflicting model ids
            if model_id in self.loaded_models:
                logger.warning(
                    f'Overwriting model {model_id} since there are duplicate entries. '
                    "Make sure you give an 'alt' id to load alternate models."
                )

            # Load model pipeline
            pipeline_msg = [f'Model: {model_id} (']

            # Load sentence segmenter
            if 'sentence_split' in model_config:
                if model_config['sentence_split'] == 'nltk':
                    pipeline_msg.append('sentence_split-nltk')
                    model['sentence_segmenter'] = nltk_sentence_segmenter
                elif type(model_config['sentence_split']) == list:
                    pipeline_msg.append('sentence_split-custom')
                    model['sentence_segmenter'] = get_custom_tokenizer(
                        model_config['sentence_split']
                    )
                else:
                    model['sentence_segmenter'] = None
            else:
                model['sentence_segmenter'] = None

            # Load pre/post-processors
            model['preprocessors'] = []
            model['postprocessors'] = []
            bpe_ok = False
            sentencepiece_ok = False

            if (
                'lowercase' in model_config['pipeline']
                and model_config['pipeline']['lowercase']
            ):
                model['preprocessors'].append(lowercaser)
                pipeline_msg.append('lowercase')

            if (
                'tokenize' in model_config['pipeline']
                and model_config['pipeline']['tokenize']
            ):
                tokenizer = get_moses_tokenizer(model_config['src'])
                model['preprocessors'].append(tokenizer)
                pipeline_msg.append('mtokenize')

            if (
                model_dir
                and 'bpe' in model_config['pipeline']
                and model_config['pipeline']['bpe']
            ):
                if not 'bpe_file' in model_config:
                    logger.warning(
                        f'Failed to load bpe model for {model_id}: '
                        'bpe_file not specified. Skipping load.'
                    )
                    continue

                model_file = os.path.join(model_dir, model_config['bpe_file'])

                if not os.path.exists(model_file):
                    logger.warning(
                        f'Failed to load bpe model for {model_id}: '
                        f'BPE vocabulary file not found at {model_file}. Skipping load.'
                    )
                    continue

                bpe_segmenter = get_bpe_segmenter(model_file)

                if not bpe_segmenter:
                    logger.warning(
                        f'Failed to loading bpe model {model_file} for {model_id}. Skipping load.'
                    )
                    continue

                model['preprocessors'].append(get_bpe_segmenter(model_file))
                bpe_ok = True
                pipeline_msg.append('bpe')
            elif (
                model_dir
                and 'sentencepiece' in model_config['pipeline']
                and model_config['pipeline']['sentencepiece']
            ):
                if not model_dir:
                    logger.warning(
                        f'Failed to load sentencepiece model for {model_id}: '
                        'model_path not specified. Skipping load.'
                    )
                    continue

                if not 'src_sentencepiece_model' in model_config:
                    logger.warning(
                        f'Failed to load sentencepiece model for {model_id}: '
                        'src_sentencepiece_model not specified. Skipping load.'
                    )
                    continue

                model_file = os.path.join(
                    model_dir, model_config['src_sentencepiece_model']
                )
                if not os.path.exists(model_file):
                    logger.warning(
                        f'Failed to load sentencepiece model for {model_id}: '
                        f'Sentencepiece model file not found at {model_file}. Skipping load.'
                    )
                    continue

                model['preprocessors'].append(
                    get_sentencepiece_segmenter(model_file)
                )
                sentencepiece_ok = True
                pipeline_msg.append('sentencepiece')
            elif model_config['model_type'] == 'ctranslator2':
                # default tokenizer needed for ctranslator2 translation
                model['preprocessors'].append(token_segmenter)

            if (
                'translate' in model_config['pipeline']
                and model_config['pipeline']['translate']
            ):
                pipeline_msg.append('translate')
                if model_config['model_type'] == 'ctranslator2':
                    if not model_dir:
                        logger.warning(
                            f'Failed to load ctranslate model for {model_id}: '
                            'model_path not specified. Skipping load.'
                        )
                        continue

                    model['translator'] = get_batch_ctranslator(model_dir)
                    pipeline_msg.append('-ctranslator2')
                elif model_config['model_type'] == 'opus':
                    opus_translator = get_batch_opustranslator(
                        model['src'], model['tgt']
                    )
                    if opus_translator:
                        model['translator'] = get_batch_opustranslator(
                            model['src'], model['tgt']
                        )
                        pipeline_msg.append('-opus-huggingface')
                    else:
                        logger.warning(
                            f'Failed to load opus-huggingface model for {model_id}. Skipping load.'
                        )
                        continue
                elif model_config['model_type'] == 'dummy':
                    pipeline_msg.append('-dummy')
                    model['translator'] = dummy_translator
            else:
                model['translator'] = None

            if bpe_ok:
                model['postprocessors'].append(desegmenter)
                pipeline_msg.append('unbpe')
            elif sentencepiece_ok:
                if not 'tgt_sentencepiece_model' in model_config:
                    logger.warning(
                        f'Failed to load sentencepiece model for {model_id}: '
                        'tgt_sentencepiece_model not specified. Skipping load.'
                    )
                    continue

                model_file = os.path.join(
                    model_dir, model_config['tgt_sentencepiece_model']
                )

                model['postprocessors'].append(
                    get_sentencepiece_desegmenter(model_file)
                )
                pipeline_msg.append('desentencepiece')
            elif model_config['model_type'] == 'ctranslator2':
                model['postprocessors'].append(token_desegmenter)

            if (
                'tokenize' in model_config['pipeline']
                and model_config['pipeline']['tokenize']
            ):
                detokenizer = get_moses_detokenizer(model_config['tgt'])
                model['postprocessors'].append(detokenizer)
                pipeline_msg.append('mdetokenize')

            if (
                'recase' in model_config['pipeline']
                and model_config['pipeline']['recase']
            ):
                model['postprocessors'].append(capitalizer)
                pipeline_msg.append('recase')

            pipeline_msg.append(')')

            logger.info(' '.join(pipeline_msg))

            # All good, add model to the list
            self.loaded_models[model_id] = model

        return 1

