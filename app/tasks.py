from celery import shared_task

from app.helpers.config import Config
from app.utils.translate import translate_text


@shared_task
def translate_text_async(model_id, text):
    config = Config(model_id=model_id, log_messages=False)
    return translate_text(model_id, text)


@shared_task
def translate_batch_async(model_id, texts):
    config = Config(model_id=model_id, log_messages=False)

    translated_batch = []
    for sentence in texts:
        translation = translate_text(model_id, sentence)
        translated_batch.append(translation)

    return translated_batch
