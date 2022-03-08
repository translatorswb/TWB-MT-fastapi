from app.utils.translate import translate_text
from app.utils.utils import get_model_id
from .base_test_case import BaseTestCase


class TestTranslations(BaseTestCase):
    def test_translate_text_en_fr(self):
        model_id = get_model_id('en', 'fr')
        text = 'Hello there, how are you doing?'
        expected_translation = 'Bonjour, comment allez-vous?'
        translation = translate_text(model_id, text)
        assert translation == expected_translation
