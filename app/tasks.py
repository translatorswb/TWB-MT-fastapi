import json

from celery import shared_task

from app.helpers.config import Config
from app.utils.translate import translate_text
from app.settings import CONFIG_JSON_PATH


@shared_task
def translate_text_async(model_id, text):
    with open(CONFIG_JSON_PATH, 'r') as f:
        conf = json.loads(f.read())
    model_data = list(
        filter(lambda x: f'{x["src"]}-{x["tgt"]}' == model_id, conf['models'])
    )
    config_data = {**conf, 'models': model_data}
    config = Config(config_data=config_data)
    return translate_text(model_id, text)
