from .base_test_case import BaseTestCase


class TestLoadModels(BaseTestCase):
    def test_load_sigle_model_with_warnings(self):
        expected_msg = 'Model: en-fr ( sentence_split-nltk lowercase translate-opus-huggingface recase )'
        assert len(self.config.messages) == 1
        assert self.config.messages[0] == expected_msg
        expected_warning = "Model path not specified for model en-fr. Can't load custom translation model or segmenters."
        assert len(self.config.warnings) == 1
        assert self.config.warnings[0] == expected_warning
