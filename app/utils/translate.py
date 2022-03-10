from typing import Optional

from app.helpers.config import Config


def translate_text(model_id: str, text: str) -> Optional[str]:
    config = Config()
    print('***** vars(config)', str(vars(config)), flush=True)


    if not model_id in config.loaded_models:
        print('***** "im here"', str("im here"), flush=True)
        return None

    if config.loaded_models[model_id]['sentence_segmenter']:
        sentence_batch = config.loaded_models[model_id]['sentence_segmenter'](
            text
        )
    else:
        sentence_batch = [text]

    # Preprocess
    for proc in config.loaded_models[model_id]['preprocessors']:
        sentence_batch = [proc(s) for s in sentence_batch]

    # Translate batch (ctranslate only)
    if config.loaded_models[model_id]['translator']:
        translated_sentence_batch = config.loaded_models[model_id][
            'translator'
        ](sentence_batch)
    else:
        translated_sentence_batch = sentence_batch

    # Postprocess
    tgt_sentences = translated_sentence_batch
    for proc in config.loaded_models[model_id]['postprocessors']:
        tgt_sentences = [proc(s) for s in tgt_sentences]

    tgt_text = ' '.join(tgt_sentences)

    return tgt_text
