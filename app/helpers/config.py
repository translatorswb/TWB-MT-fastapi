import json
import logging
import os

from app.constants import SUPPORTED_MODEL_TYPES
from app.exceptions import ConfigurationException, ModelLoadingException
from app.helpers.singleton import Singleton
from app.settings import (
    CONFIG_JSON_PATH,
    MODELS_ROOT_DIR,
)
from app.utils.pipeline import pipeline
from app.utils.utils import (
    get_model_id,
    parse_model_id,
)

logger = logging.getLogger('console_logger')


class Config(metaclass=Singleton):
    def __init__(self, config_file=None, config_data=None):
        self.loaded_models = {}
        self.language_codes = {}
        self.languages_list = {}
        self.config_data = config_data or {}
        self.config_file = config_file or CONFIG_JSON_PATH

        self.warnings = []
        self.messages = []

        self._validate()
        self._load_all_models()
        self._load_languages_list()

    def map_lang_to_closest(self, lang):
        if lang in self.language_codes:
            return lang
        elif '_' in lang:
            superlang = lang.split('_')[0]
            if superlang in self.language_codes:
                return superlang
        return ''

    def _get_model_path(self, model_config, model_id):
        model_dir = None

        # Check model path
        if 'model_path' in model_config and model_config['model_path']:
            model_dir = os.path.join(
                MODELS_ROOT_DIR, model_config['model_path']
            )
            if not os.path.exists(model_dir):
                model_dir = None
                self._log_warning(
                    f'Model path {model_dir} not found for model {model_id}. '
                    "Can't load custom translation model or segmenters."
                )
        else:
            self._log_warning(
                f'Model path not specified for model {model_id}. '
                "Can't load custom translation model or segmenters."
            )
        return model_dir

    def _is_valid_model_config(self, model_config):
        # Check if model_type src and tgt fields are specified
        for item in ['src', 'tgt', 'model_type']:
            if item not in model_config:
                self._log_warning(
                    f'`{item}` not speficied for a model. Skipping load'
                )
                return False
        return True

    def _is_valid_model_type(self, model_type):
        if not model_type in SUPPORTED_MODEL_TYPES:
            self._log_warning(
                f'`model_type` not recognized: {model_type}. Skipping load'
            )
            return False
        return True

    def _load_all_models(self):
        for model_config in self.config_data['models']:
            if not 'load' in model_config or not model_config['load']:
                continue

            # CONFIG CHECKS
            if not self._is_valid_model_config(model_config):
                continue

            if not self._is_valid_model_type(model_config['model_type']):
                continue

            try:
                self._load_model(model_config)
            except ModelLoadingException:
                continue

    def _load_model(self, model_config):
        src = model_config['src']
        tgt = model_config['tgt']
        alt_id = model_config.get('alt')
        pipeline_msg = []
        model_id = get_model_id(
            model_config['src'], model_config['tgt'], alt_id
        )
        model_dir = self._get_model_path(model_config, model_id)
        model = {
            'src': src,
            'tgt': tgt,
            'sentence_segmenter': None,
            'preprocessors': [],
            'postprocessors': [],
        }
        checks = {
            'bpe_ok': False,
            'sentencepiece_ok': False,
        }
        kwargs = {
            'model_config': model_config,
            'model': model,
            'checks': checks,
            'pipeline_msg': pipeline_msg,
            'model_dir': model_dir,
            'model_id': model_id,
            'warn': self._log_warning,
        }

        # Check if language names exist for the language ids
        self._validate_src_tgt(src, tgt)

        # More configuration checks
        self._validate_model_conflicts(model_config, model_id)

        # Load model pipeline
        for loader in pipeline:
            loader(**kwargs)

        self._log_info(f"Model: {model_id} ( {' '.join(pipeline_msg)} )")

        # All good, add model to the list
        self.loaded_models[model_id] = model

    def _load_languages_list(self):
        for model_id in self.loaded_models.keys():
            source, target, alt = parse_model_id(model_id)
            if not source in self.languages_list:
                self.languages_list[source] = {}
            if not target in self.languages_list[source]:
                self.languages_list[source][target] = []

            self.languages_list[source][target].append(model_id)

    def _log_warning(self, msg):
        logger.warning(msg)
        self.warnings.append(msg)

    def _log_info(self, msg):
        logger.info(msg)
        self.messages.append(msg)

    def _validate(self):
        self._validate_config_file()
        self._validate_models()
        self._validate_languages()

    def _validate_config_file(self):
        if self.config_data:
            return

        # Check if config file is there and well formatted
        if not os.path.exists(self.config_file):
            self._log_warning(
                f'Config file {self.config_file} not found. '
                'No models will be loaded.'
            )
        else:
            try:
                with open(self.config_file, 'r') as jsonfile:
                    self.config_data = json.load(jsonfile)
            except json.decoder.JSONDecodeError:
                msg = 'Config file format broken. No models will be loaded.'
                logger.error(msg)
                raise ConfigurationException(msg)

    def _validate_models(self):
        # Check if MODELS_ROOT_DIR exists
        if not os.path.exists(MODELS_ROOT_DIR):
            msg = '`models` directory not found. No models will be loaded.'
            logger.error(msg)
            raise ConfigurationException(msg)

        if not 'models' in self.config_data:
            msg = (
                "Model spefication list ('models') not found in configuration."
            )
            logger.error(msg)
            raise ConfigurationException(msg)

    def _validate_languages(self):
        if 'languages' in self.config_data:
            self.language_codes = self.config_data['languages']
            logger.debug(f'Languages: {self.language_codes}')
        else:
            self._log_warning(
                "Language name spefication dictionary ('languages') not found in configuration."
            )

    def _validate_src_tgt(self, src, tgt):
        if not src in self.language_codes:
            self._log_warning(
                f'Source language code `{src}` not defined in '
                'languages dict. This will surely break something.'
            )
        if not tgt in self.language_codes:
            self._log_warning(
                f'Target language code `{tgt}` not defined in '
                'languages dict. This will surely break something.'
            )

    def _validate_model_conflicts(self, model_config, model_id):
        # Check conflicting subword segmenters
        if (
            'bpe' in model_config['pipeline']
            and 'sentencepiece' in model_config['pipeline']
        ):
            self._log_warning(
                f'Model {model_id} has both sentencepiece and bpe setup. '
            )

        # Check conflicting model ids
        if model_id in self.loaded_models:
            self._log_warning(
                f'Overwriting model {model_id} since there are duplicate entries. '
                'Make sure you give an `alt` id to load alternate models.'
            )
