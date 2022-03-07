from app.helpers.config import Config
from app.utils.translate import translate_text
from app.utils.utils import get_model_id


class TestTranslations:
    def setup(self):
        self.config_data = {
            'languages': {
                'en': 'English',
                'fr': 'French',
            },
            'models': [
                {
                    'src': 'en',
                    'tgt': 'fr',
                    'model_type': 'opus',
                    'load': True,
                    'sentence_split': 'nltk',
                    'pipeline': {
                        'lowercase': True,
                        'translate': True,
                        'recase': True,
                    },
                },
            ],
        }
        self.config = Config(config_data=self.config_data)
        self.model_id = get_model_id('en', 'fr')

    def test_translate_text_en_fr(self):
        text = 'Hello there, how are you doing?'
        expected_translation = 'Bonjour, comment allez-vous?'
        translation = translate_text(self.model_id, text)
        assert translation == expected_translation
