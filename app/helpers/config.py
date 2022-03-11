import json
import logging
import os
from typing import Optional, Dict, List

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
    def __init__(
        self,
        config_file: Optional[str] = None,
        config_data: Optional[Dict] = None,
        load_all_models: bool = False,
    ):
        self.loaded_models: Dict = {}
        self.language_codes: Dict = {}
        self.languages_list: Dict = {}
        self.config_data: Dict = config_data or {}
        self.config_file: str = config_file or CONFIG_JSON_PATH
        self.load_all_models: bool = load_all_models

        self.warnings: List[str] = []
        self.messages: List[str] = []

        if not config_data:
            self._validate()

        if self.load_all_models or config_data:
            self._load_language_codes()
            self._load_all_models()
            self._load_languages_list()

    def map_lang_to_closest(self, lang: str) -> str:
        if lang in self.language_codes:
            return lang
        elif '_' in lang:
            superlang = lang.split('_')[0]
            if superlang in self.language_codes:
                return superlang
        return ''

    def _get_model_path(
        self, model_config: Dict, model_id: str
    ) -> Optional[str]:
        model_dir = None

        # Check model path
        if 'model_path' in model_config and model_config['model_path']:
            model_dir = os.path.join(
                MODELS_ROOT_DIR, model_config['model_path']
            )
            if not os.path.exists(model_dir):
                self._log_warning(
                    f'Model path {model_dir} not found for model {model_id}. '
                    "Can't load custom translation model or segmenters."
                )
                model_dir = None
        else:
            self._log_warning(
                f'Model path not specified for model {model_id}. '
                "Can't load custom translation model or segmenters."
            )
        return model_dir

    def _is_valid_model_config(self, model_config: Dict) -> bool:
        # Check if model_type src and tgt fields are specified
        for item in ['src', 'tgt', 'model_type']:
            if item not in model_config:
                self._log_warning(
                    f'`{item}` not speficied for a model. Skipping load'
                )
                return False
        return True

    def _is_valid_model_type(self, model_type: str) -> bool:
        if not model_type in SUPPORTED_MODEL_TYPES:
            self._log_warning(
                f'`model_type` not recognized: {model_type}. Skipping load'
            )
            return False
        return True

    def _load_all_models(self) -> None:
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

    def _load_model(self, model_config: Dict) -> None:
        src: str = model_config['src']
        tgt: str = model_config['tgt']
        alt_id: Optional[str] = model_config.get('alt')
        pipeline_msg: List[str] = []
        model_id: str = get_model_id(src, tgt, alt_id)
        model_dir: Optional[str] = self._get_model_path(model_config, model_id)
        model: Dict = {
            'src': src,
            'tgt': tgt,
            'sentence_segmenter': None,
            'preprocessors': [],
            'postprocessors': [],
        }
        checks: Dict = {
            'bpe_ok': False,
            'sentencepiece_ok': False,
        }
        kwargs: Dict = {
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

    def _load_language_codes(self) -> None:
        if 'languages' in self.config_data:
            self.language_codes = self.config_data['languages']
            logger.debug(f'Languages: {self.language_codes}')
        else:
            self._log_warning(
                "Language name spefication dictionary ('languages') not found in configuration."
            )

    def _load_languages_list(self) -> None:
        for model_id in self.loaded_models.keys():
            if not (parsed_id := parse_model_id(model_id)):
                self._log_warning(f'Unable to parse model_id of {model_id}')
                continue
            source, target, alt = parsed_id
            if not source in self.languages_list:
                self.languages_list[source] = {}
            if not target in self.languages_list[source]:
                self.languages_list[source][target] = []

            self.languages_list[source][target].append(model_id)

    def _log_warning(self, msg: str) -> None:
        logger.warning(msg)
        self.warnings.append(msg)

    def _log_info(self, msg: str) -> None:
        logger.info(msg)
        self.messages.append(msg)

    def _validate(self) -> None:
        self._validate_config_file()
        self._validate_models()

    def _validate_config_file(self) -> None:
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

    def _validate_models(self) -> None:
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

    def _validate_src_tgt(self, src: str, tgt: str) -> None:
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

    def _validate_model_conflicts(
        self, model_config: Dict, model_id: str
    ) -> None:
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
