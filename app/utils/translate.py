from typing import Optional
import logging
import os
from app.helpers.config import Config
from app.utils.utils import parse_model_id, get_model_id
from app.constants import MULTIMODALCODE

DEVDEBUG = True
logger = logging.getLogger('console_logger')

# TODO: This should get text batch
def translate_text(model_id: str, text: str, src: str, tgt: str) -> Optional[str]:    
    config = Config()
    if DEVDEBUG: logger.debug(f'translate.py/translate_text for {model_id} {src}->{tgt} | {text}') #print('>translate.py/translate_text for', model_id, text, src, tgt)

    

    model = config.loaded_models[model_id]

    if model['sentence_segmenter']:
        sentence_batch = model['sentence_segmenter'](
            text
        )
    else:
        sentence_batch = [text]

    if DEVDEBUG: logger.debug(f'>translate_text:sentence_batch {sentence_batch}')

    # Pre-translate
    if model['pretranslatechain']:
        for pair in model['pretranslatechain']:
            chainmodel_src, chainmodel_tgt, chainmodel_alt = parse_model_id(pair)
            if not pair in config.loaded_models:
                pair = get_model_id(MULTIMODALCODE, MULTIMODALCODE) #TODO: Not tested with alt
            sentence_batch = config.loaded_models[pair]['translator'](sentence_batch, chainmodel_src, chainmodel_tgt)
            if DEVDEBUG: logger.debug(f'>translate_text:Pre-translate {pair}, {chainmodel_src}-{chainmodel_tgt} {sentence_batch}')
    
    # Preprocess
    for proc in model['preprocessors']:
        sentence_batch = [proc(s) for s in sentence_batch]
        if DEVDEBUG: logger.debug(f'>translate_text:Preprocess/sentence_batch {sentence_batch}')
    
    # Translate batch
    if model['translator']:
        translated_sentence_batch = []
        if model['model_type'] == 'nllb':
            # for sentence in sentence_batch:
            inference_url = "/predictions/nllb-200-1.3B"
            status, translated_sentence = model[
                'translator'
            ](inference_url, sentence_batch, src, tgt)
            if status == 'success':
                translated_sentence_batch = translated_sentence
            else:
                print('##########inference failure##########')

        if model['model_type'] == 'opus':
            inference_url = f'/predictions/opus-mt-{src}-{tgt}'
            # for sentence in sentence_batch:     
            status, translated_sentence = model[
                'translator'
            ](inference_url, sentence_batch, src, tgt)
            if status == 'success':
                translated_sentence_batch = translated_sentence
            else:
                print('##########inference failure##########')


        # translated_sentence_batch = model[
        #     'translator'
        # ](sentence_batch, src, tgt)
        if DEVDEBUG: logger.debug(f'>translate_text:Translate batch /translated_sentence_batch {translated_sentence_batch}')
    else:
        translated_sentence_batch = sentence_batch
        if DEVDEBUG: logger.debug(f'>translate_text:else Translate batch /translated_sentence_batch {translated_sentence_batch}')
    
    # Postprocess
    tgt_sentences = translated_sentence_batch
    for proc in model['postprocessors']:
        tgt_sentences = [proc(s) for s in tgt_sentences]
    if DEVDEBUG: logger.debug(f'>translate_text:tgt_sentences {tgt_sentences}')
    
    # Post-translate
    if model['posttranslatechain']:
        for pair in model['posttranslatechain']:
            chainmodel_src, chainmodel_tgt, chainmodel_alt = parse_model_id(pair)
            if not pair in config.loaded_models:
                pair = get_model_id(MULTIMODALCODE, MULTIMODALCODE) #TODO: Not tested with alt
            tgt_sentences = config.loaded_models[pair]['translator'](tgt_sentences, chainmodel_src, chainmodel_tgt)
            if DEVDEBUG: logger.debug(f'>translate_text:Post-translate {pair}, {chainmodel_src}-{chainmodel_tgt} {tgt_sentences}')
    tgt_text = ' '.join(tgt_sentences)

    return tgt_text
