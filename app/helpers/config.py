import json
import logging
import os
from typing import Optional, Dict, List

from app.constants import SUPPORTED_MODEL_TYPES, MULTIMODALCODE, MODEL_TAG_SEPARATOR
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
        if '_' in lang:
            superlang = lang.split('_')[0]
            if superlang in self.language_codes:
                return superlang
        return lang

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

    def _get_pretranslators(
        self, model_config: Dict, model_id: str
    ) -> Optional[str]:
        pretranslator_chain = []

        # Check model path
        if 'pretranslatechain' in model_config and model_config['pretranslatechain']:
            #check if all pairs are in config
            for pair in model_config['pretranslatechain']:
                pair_found = False
                for m in self.config_data['models']:
                    #TODO
                    if 'multilingual' in m and m['multilingual']:
                        continue
                    m_id = get_model_id(src=m['src'], tgt=m['tgt'])
                    if m_id == pair and m['load']:
                        pair_found = True
                        break
                if not pair_found: 
                    self._log_warning(
                        f'Pretranslation model {pair} not found or is not active. '
                        f'Can\'t load pretranslator chain for model {model_id}'
                    )
                    return pretranslator_chain

            pretranslator_chain = model_config['pretranslatechain']

        return pretranslator_chain

    def _get_posttranslators(
        self, model_config: Dict, model_id: str
    ) -> Optional[str]:
        posttranslator_chain = []

        # Check model path
        if 'posttranslatechain' in model_config and model_config['posttranslatechain']:
            #check if all pairs are in config
            for pair in model_config['posttranslatechain']:
                pair_found = False
                for m in self.config_data['models']:
                    #TODO
                    if 'multilingual' in m and m['multilingual']:
                        continue
                    m_id = get_model_id(src=m['src'], tgt=m['tgt'])
                    if m_id == pair and m['load']:
                        pair_found = True
                        break
                if not pair_found: 
                    self._log_warning(
                        f'Posttranslation model {pair} not found or is not active. '
                        f'Can\'t load posttranslator chain for model {model_id}'
                    )
                    return posttranslator_chain

            posttranslator_chain = model_config['posttranslatechain']

        return posttranslator_chain

    def _is_valid_model_config(self, model_config: Dict) -> bool:
        # Check if model_type src and tgt fields are specified
        if 'model_type' not in model_config:
            self._log_warning(
                    f'`{item}` not speficied for a model. Skipping load'
                )
            return False
        if 'multilingual' not in model_config or not model_config['multilingual']:
            for item in ['src', 'tgt']:
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

        self._log_info(f'{len(self.loaded_models)} models ' + str(list(self.loaded_models.keys())))

    def _load_model(self, model_config: Dict) -> None:
        model_type: str = model_config.get('model_type')
        src: Optional[str] = model_config['src'] if 'src' in model_config else MULTIMODALCODE
        tgt: Optional[str] = model_config['tgt'] if 'src' in model_config else MULTIMODALCODE
        alt_id: Optional[str] = model_config.get('alt')
        multilingual: Optional[bool] = model_config['multilingual'] if 'multilingual' in model_config else False
        supported_pairs: List[str] = model_config['supported_pairs'] if 'supported_pairs' in model_config else []
        pipeline_msg: List[str] = []
        model_id: str = get_model_id(src=src, tgt=tgt, alt_id=alt_id)
        model_dir: Optional[str] = self._get_model_path(model_config, model_id)
        pretranslatechain: List[str] = self._get_pretranslators(model_config, model_id)
        posttranslatechain: List[str] = self._get_posttranslators(model_config, model_id)
        model: Dict = {
            'model_type': model_type,
            'multilingual': multilingual,
            'supported_pairs': supported_pairs,
            'src': src,
            'tgt': tgt,
            'sentence_segmenter': None,
            'pretranslatechain': pretranslatechain,
            'posttranslatechain': posttranslatechain,
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
            self.language_codes[MULTIMODALCODE] = "Multilingual"
            self._log_info(f'Language names: {self.language_codes}')
        else:
            self._log_warning(
                "Language name spefication dictionary ('languages') not found in configuration."
            )

    def _load_languages_list(self) -> None:
        for main_model_id in self.loaded_models.keys():
            main_parsed_id = parse_model_id(main_model_id)
            if not (main_parsed_id := parse_model_id(main_model_id)):
                self._log_warning(f'Unable to parse model_id of {main_model_id}')
                continue

            source_main, target_main, alt_main = main_parsed_id

            models_to_add = [] #(model_id, source, target, alt)

            if self.loaded_models[main_model_id]['multilingual']:
                for model_id in self.loaded_models[main_model_id]['supported_pairs']:
                    parsed_id = parse_model_id(model_id)
                    if not (parsed_id := parse_model_id(model_id)):
                        self._log_warning(f'Unable to parse multilingual model pair {model_id} of {main_model_id}')
                        continue
                    source, target, alt = parsed_id

                    if alt_main:
                        multimodel_code = MULTIMODALCODE + MODEL_TAG_SEPARATOR + model_id + MODEL_TAG_SEPARATOR + alt_main
                    else:
                        multimodel_code = MULTIMODALCODE + MODEL_TAG_SEPARATOR + model_id

                    models_to_add.append((multimodel_code, source, target, alt))
            else:
                models_to_add.append((main_model_id, source_main, target_main, alt_main))

            for model_info in models_to_add:
                model_id, source, target, alt = model_info
                if not source in self.languages_list:
                    self.languages_list[source] = {}
                if not target in self.languages_list[source]:
                    self.languages_list[source][target] = []

                self.languages_list[source][target].append(model_id)

        self._log_info(f'Languages list: {self.languages_list}')

    def _lookup_pair_in_languages_list(self, src, tgt, alt=None):
        if src in self.languages_list:
            if tgt in self.languages_list[src]:
                if self.languages_list[src][tgt]:
                    if alt:
                        return [mid for mid in self.languages_list[src][tgt] if mid.endswith(alt)]
                    else:
                        return self.languages_list[src][tgt]
        return []

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
            # raise ConfigurationException(msg)

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
                'languages dict. This might break something.'
            )
        if not tgt in self.language_codes:
            self._log_warning(
                f'Target language code `{tgt}` not defined in '
                'languages dict. This might break something.'
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
