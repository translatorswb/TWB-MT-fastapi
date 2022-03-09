from .base_test_case import BaseTestCase


class TestLoadModels(BaseTestCase):
    def test_load_sigle_model_with_warnings(self):
        # messages
        expected_msg = 'Model: en-fr ( sentence_split-nltk lowercase translate-opus-huggingface recase )'
        assert len(self.config.messages) == 1
        assert self.config.messages[0] == expected_msg
        expected_warning = "Model path not specified for model en-fr. Can't load custom translation model or segmenters."
        assert len(self.config.warnings) == 1
        assert self.config.warnings[0] == expected_warning

        # languages
        assert self.config.language_codes == {'en': 'English', 'fr': 'French'}
        assert self.config.languages_list == {'en': {'fr': ['en-fr']}}

        # model
        assert len(self.config.loaded_models) == 1
        assert 'en-fr' in self.config.loaded_models
        model = self.config.loaded_models['en-fr']
        assert model['src'] == 'en'
        assert model['tgt'] == 'fr'

        # pipeline
        assert model['sentence_segmenter'].__name__ == 'nltk_sentence_segmenter'
        assert len(model['preprocessors']) == 1
        assert model['preprocessors'][0].__name__ == 'lowercaser'
        assert len(model['postprocessors']) == 1
        assert model['postprocessors'][0].__name__ == 'capitalizer'
        assert 'get_batch_opustranslator' in str(model['translator'])
