from celery import shared_task

from app.utils.translate import translate_text
from app.helpers.config import Config

@shared_task
def translate_text_async(model_id, text):
    t = translate_text(model_id, text)
