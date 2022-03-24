from app.helpers.config import Config
from app.tasks import translate_batch_async, translate_text_async


def test_task_translate_text_async():
    options = {
        'model_id': 'en-fr',
        'text': 'Hello there, how are you doing?',
    }
    expected_translation = 'Bonjour, comment allez-vous?'
    translation = translate_text_async(**options)
    assert translation == expected_translation

def test_task_translate_batch_async():
    options = {
        'model_id': 'en-fr',
        'texts': ['Hello, what is your name?', 'How are you doing?'],
    }
    expected_translations = [
        'Bonjour, quel est votre nom?',
        'Comment ça va?',
    ]
    translation = translate_batch_async(**options)
    assert translation == expected_translations
