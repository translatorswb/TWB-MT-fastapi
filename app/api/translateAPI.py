import json
import os
import re
from typing import List, Optional, Dict, Callable

from fastapi import Header, APIRouter, HTTPException
from nltk.tokenize import sent_tokenize, word_tokenize
from pydantic import BaseModel

from app.constants import (
    CONFIG_JSON_PATH,
    CTRANSLATE_DEVICE,
    CTRANSLATE_INTER_THREADS,
    HELSINKI_NLP,
    MODELS_ROOT_DIR,
    MODEL_TAG_SEPARATOR,
    MOSES_TOKENIZER_DEFAULT_LANG,
    SUPPORTED_MODEL_TYPES,
)
from app.utils.translators import (
    get_batch_ctranslator,
    get_batch_opustranslator,
    get_ctranslator,
    dummy_translator,
)
from app.utils.segmenters import (
    desegmenter,
    nltk_sentence_segmenter,
    token_desegmenter,
    token_segmenter,
    get_bpe_segmenter,
    get_sentencepiece_segmenter,
    get_sentencepiece_desegmenter,
)
from app.utils.utils import (
    capitalizer,
    get_model_id,
    lowercaser,
    parse_model_id,
)
from app.utils.tokenizers import (
    get_custom_tokenizer,
    get_moses_detokenizer,
    get_moses_tokenizer,
)

translate = APIRouter(prefix='/api/v1/translate')

#models and data
loaded_models = {}
config_data = {}
language_codes = {}


def map_lang_to_closest(lang):
    global language_codes
    if lang in language_codes:
        return lang
    elif '_' in lang:
        superlang = lang.split('_')[0]
        if superlang in language_codes:
            return superlang
    return ''


def do_translate(model_id, text):
    # print("do_translate")
    # print(text)

    if model_id in loaded_models:
        if loaded_models[model_id]['sentence_segmenter']:
            sentence_batch = loaded_models[model_id]['sentence_segmenter'](text)
        else:
            sentence_batch = [text]

        # print("sentence_batch")
        # print(sentence_batch)

        #preprocess
        # print("Preprocess")
        for step in loaded_models[model_id]['preprocessors']:
            sentence_batch = [step(s) for s in sentence_batch]
            # print(step)
            # print(sentence_batch)

        #translate batch (ctranslate only)
        if loaded_models[model_id]['translator']:
            translated_sentence_batch = loaded_models[model_id]['translator'](sentence_batch)
            # print("translated_sentence_batch")
            # print(translated_sentence_batch)
        else:
            translated_sentence_batch = sentence_batch

        #postprocess
        # print("Postprocess")
        tgt_sentences = translated_sentence_batch
        for step in loaded_models[model_id]['postprocessors']:
            tgt_sentences = [step(s) for s in tgt_sentences]
            # print(step)
            # print(tgt_sentences)

        tgt_text = " ".join(tgt_sentences)
        # print("tgt_text")
        # print(tgt_text)

        return tgt_text
    else:
        return 0

#Functional
# def read_config(config_file):
#     with open(config_file, "r") as jsonfile: 
#         data = json.load(jsonfile) 
#         print("Config Read successful") 
#         print(data)
#     return data

def load_models(config_path):
    #Check if config file is there and well formatted
    if not os.path.exists(CONFIG_JSON_PATH):
        print("WARNING: Config file %s not found. No models will be loaded."%CONFIG_JSON_PATH)
        return 0

    try:
        with open(CONFIG_JSON_PATH, "r") as jsonfile: 
            config_data = json.load(jsonfile)
    except:
        print("ERROR: Config file format broken. No models will be loaded.")
        return 0

    #Check if MODELS_ROOT_DIR exists
    if not os.path.exists(MODELS_ROOT_DIR):
        print("ERROR: models directory not found. No models will be loaded.")
        return 0

    if 'languages' in config_data:
        global language_codes
        language_codes = config_data['languages']
        print("Languages: %s"%language_codes)
    else:
        print("WARNING: Language name spefication dictionary ('languages') not found in configuration." )

    if not 'models' in config_data:
        print("ERROR: Model spefication list ('models') not found in configuration." )
        return 0

    for model_config in config_data['models']:
        if not 'load' in model_config or model_config['load']:
            #CONFIG CHECKS
            #Check if model_type src and tgt fields are specified
            if not 'src' in model_config:
                print("WARNING: Source language (src) not speficied for a model. Skipping load")
                continue

            if not 'tgt' in model_config:
                print("WARNING: Target language (tgt) not speficied for a model. Skipping load")
                continue

            if not 'model_type' in model_config:
                print("WARNING: model_type not speficied for model. Skipping load")
                continue

            if not model_config['model_type'] in SUPPORTED_MODEL_TYPES:
                print("WARNING: model_type not recognized: %s. Skipping load"%model_config['model_type'])
                continue

            #Load model variables
            model = {}
            model['src'] = model_config['src']
            model['tgt'] = model_config['tgt']
            
            if 'alt' in model_config:
                alt_id = model_config['alt']
            else:
                alt_id = None

            model_id = get_model_id(model_config['src'], model_config['tgt'], alt_id)

            #Check if language names exist for the language ids
            if not model['src'] in language_codes:
                print("WARNING: Source language code %s not defined in languages dict. This will surely break something."%model['src'])
            if not model['tgt'] in language_codes:
                print("WARNING: Target language code %s not defined in languages dict. This will surely break something."%model['tgt'])

            #Check model path
            if 'model_path' in model_config and model_config['model_path']:
                model_dir = os.path.join(MODELS_ROOT_DIR, model_config['model_path'])
                if not os.path.exists(model_dir):
                    print("WARNING: Model path %s not found for model %s. Can't load custom translation model or segmenters."%(model_dir, model_id))
                    model_dir = None
            else:
                print("WARNING: Model path not specified for model %s. Can't load custom translation model or segmenters."%model_id)
                model_dir = None

            #More configuration checks
            #Check conflicting subword segmenters
            if 'bpe' in model_config['pipeline'] and 'sentencepiece' in model_config['pipeline']:
                print("WARNING: Model %s has both sentencepiece and bpe setup. "%model_id)

            #Check conflicting model ids
            if model_id in loaded_models:
                print("WARNING: Overwriting model %s since there are duplicate entries. Make sure you give an 'alt' ids to load alternate models."%model_id)


            #Load model pipeline
            print("Model: %s ("%model_id, end=" ")

            #Load sentence segmenter
            if 'sentence_split' in model_config:
                if model_config['sentence_split'] == "nltk":
                    print("sentence_split-nltk", end=" ")
                    model['sentence_segmenter'] = nltk_sentence_segmenter
                elif type(model_config['sentence_split']) == list:
                    print("sentence_split-custom", end=" ")
                    model['sentence_segmenter'] = get_custom_tokenizer(model_config['sentence_split'])
                else:
                    model['sentence_segmenter'] = None
            else:
                model['sentence_segmenter'] = None

            #Load pre/post-processors
            model['preprocessors'] = []
            model['postprocessors'] = []
            bpe_ok = False
            sentencepiece_ok = False

            if 'lowercase' in model_config['pipeline'] and model_config['pipeline']['lowercase']:
                model['preprocessors'].append(lowercaser)
                print("lowercase", end=" ")

            if 'tokenize' in model_config['pipeline'] and model_config['pipeline']['tokenize']:
                tokenizer = get_moses_tokenizer(model_config['src'])
                model['preprocessors'].append(tokenizer)
                print("mtokenize", end=" ")

            if model_dir and 'bpe' in model_config['pipeline'] and model_config['pipeline']['bpe']:
                if not 'bpe_file' in model_config:
                    print("\nWARNING: Failed to load bpe model for %s: bpe_file not specified. Skipping load."%(model_id))
                    continue

                model_file = os.path.join(model_dir, model_config['bpe_file'])

                if not os.path.exists(model_file):
                    print("\nWARNING: Failed to load bpe model for %s: BPE vocabulary file not found at %s. Skipping load."%(model_id, model_file))
                    continue

                bpe_segmenter = get_bpe_segmenter(model_file)

                if not bpe_segmenter:
                    print("\nWARNING: Failed to loading bpe model %s for %s. Skipping load."%(model_file, model_id))
                    continue

                model['preprocessors'].append(get_bpe_segmenter(model_file))
                bpe_ok = True
                print("bpe", end=" ")
            elif model_dir and 'sentencepiece' in model_config['pipeline'] and model_config['pipeline']['sentencepiece']:
                if not model_dir:
                    print("\nWARNING: Failed to load sentencepiece model for %s: model_path not specified. Skipping load."%(model_id))
                    continue

                if not 'src_sentencepiece_model' in model_config:
                    print("\nWARNING: Failed to load sentencepiece model for %s: src_sentencepiece_model not specified. Skipping load."%(model_id))
                    continue

                model_file = os.path.join(model_dir, model_config['src_sentencepiece_model'])
                if not os.path.exists(model_file):
                    print("\nWARNING: Failed to load sentencepiece model for %s: Sentencepiece model file not found at %s. Skipping load."%(model_id, model_file))
                    continue

                model['preprocessors'].append(get_sentencepiece_segmenter(model_file))
                sentencepiece_ok = True
                print("sentencepiece", end=" ")
            elif model_config['model_type'] == 'ctranslator2':
                #default tokenizer needed for ctranslator2 translation
                model['preprocessors'].append(token_segmenter)

            
            if 'translate' in model_config['pipeline'] and model_config['pipeline']['translate']:
                print("translate", end="")
                if model_config['model_type'] == 'ctranslator2':
                    if not model_dir:
                        print("\nWARNING: Failed to load ctranslate model for %s: model_path not specified. Skipping load."%(model_id))
                        continue

                    model['translator'] = get_batch_ctranslator(model_dir)  
                    print("-ctranslator2", end=" ") 
                elif model_config['model_type'] == 'opus':
                    opus_translator = get_batch_opustranslator(model['src'], model['tgt'])
                    if opus_translator:
                        model['translator'] = get_batch_opustranslator(model['src'], model['tgt'])
                        print("-opus-huggingface", end=" ")
                    else: 
                        print("\nWARNING: Failed to load opus-huggingface model for %s. Skipping load."%(model_id))
                        continue
                elif model_config['model_type'] == 'dummy':
                    print("-dummy", end=" ")
                    model['translator'] = dummy_translator
            else:
                model['translator'] = None

            if bpe_ok:
                model['postprocessors'].append(desegmenter)
                print("unbpe", end=" ")
            elif sentencepiece_ok:
                if not 'tgt_sentencepiece_model' in model_config:
                    print("\nWARNING: Failed to load sentencepiece model for %s: tgt_sentencepiece_model not specified. Skipping load."%(model_id))
                    continue

                model_file = os.path.join(model_dir, model_config['tgt_sentencepiece_model'])
                
                model['postprocessors'].append(get_sentencepiece_desegmenter(model_file))
                print("desentencepiece", end=" ")
            elif model_config['model_type'] == 'ctranslator2':
                model['postprocessors'].append(token_desegmenter)

            if 'tokenize' in model_config['pipeline'] and model_config['pipeline']['tokenize']:
                detokenizer = get_moses_detokenizer(model_config['tgt'])
                model['postprocessors'].append(detokenizer)
                print("mdetokenize", end=" ")

            if 'recase' in model_config['pipeline'] and model_config['pipeline']['recase']:
                model['postprocessors'].append(capitalizer)
                print("recase", end=" ")

            print(")")

            #All good, add model to the list
            loaded_models[model_id] = model
        
    return 1

    
#HTTP operations
class TranslationRequest(BaseModel):
    src: str
    tgt: str
    alt: Optional[str] = None
    text: str

class BatchTranslationRequest(BaseModel):
    src: str
    tgt: str
    alt: Optional[str] = None
    texts: List[str]

class TranslationResponse(BaseModel):
    translation: str

class BatchTranslationResponse(BaseModel):
    translation: List[str]

class LanguagesResponse(BaseModel):
    models: Dict
    languages: Dict

@translate.post('/', status_code=200)
async def translate_sentence(request: TranslationRequest):

    model_id = get_model_id(map_lang_to_closest(request.src), map_lang_to_closest(request.tgt), request.alt)

    if not model_id in loaded_models:
        raise HTTPException(status_code=404, detail="Language pair %s is not supported."%model_id)
    
    translation = do_translate(model_id, request.text)

    response = TranslationResponse(translation=translation)
    return response

@translate.post('/batch', status_code=200)
async def translate_batch(request: BatchTranslationRequest):
    print(request.texts)
    print(type(request.texts))

    model_id = get_model_id(map_lang_to_closest(request.src), map_lang_to_closest(request.tgt), request.alt)

    if not model_id in loaded_models:
        raise HTTPException(status_code=404, detail="Language pair %s is not supported."%model_id)
    
    translated_batch = []
    for sentence in request.texts:
        translation = do_translate(model_id, sentence)
        translated_batch.append(translation)

    response = BatchTranslationResponse(translation=translated_batch)
    return response

@translate.get('/', status_code=200)
async def languages():
    languages_list = {}
    for model_id in loaded_models.keys():
        source, target, alt = parse_model_id(model_id)
        if not source in languages_list:
            languages_list[source] = {}
        if not target in languages_list[source]:
            languages_list[source][target] = []

        languages_list[source][target].append(model_id)

    return LanguagesResponse(languages=language_codes, models=languages_list)
