from typing import List, Optional
from fastapi import Header, APIRouter, HTTPException
import os
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from pydantic import BaseModel 

#constants
CONFIG_JSON_PATH = os.getenv('MT_API_CONFIG') 
CTRANSLATE_DEVICE = os.getenv('MT_API_DEVICE') or 'cpu'
CTRANSLATE_INTER_THREADS = int(os.getenv('MT_API_THREADS') or '16')
MOSES_TOKENIZER_DEFAULT_LANG = 'en'

translate = APIRouter()

#models and data
loaded_models = {}
config_data = {}

#processors
nltk_sentence_segmenter = lambda x : sent_tokenize(x)   #string IN -> list OUT
lowercaser =  lambda x: x.lower() #string IN -> string OUT
desegmenter = lambda x: re.sub('(@@ )|(@@ ?$)', '', ' '.join(x)) #list IN -> string OUT
capitalizer = lambda x: x.capitalize() #string IN -> string OUT
token_segmenter = lambda x: x.strip().split()  #string IN -> list OUT
token_desegmenter = lambda x: ' '.join(x) #list IN -> string OUT
dummy_translator = lambda x: x

#MT operations
def get_model_id(src, tgt, alt_id=None):
    model_id = src + "_" + tgt
    if alt_id:
        model_id += "_" + alt_id
    return model_id

def get_ctranslator(ctranslator_model_path):
    from ctranslate2 import Translator
    ctranslator = Translator(ctranslator_model_path)
    translator = lambda x: ctranslator.translate_batch([x])[0][0]['tokens']  #list IN -> list OUT
    return translator

def get_batch_ctranslator(ctranslator_model_path): 
    from ctranslate2 import Translator
    ctranslator = Translator(ctranslator_model_path, device=CTRANSLATE_DEVICE, inter_threads=CTRANSLATE_INTER_THREADS)
    translator = lambda x: [s[0]['tokens'] for s in ctranslator.translate_batch(x)] 
    return translator

def get_batch_opustranslator(src, tgt):   
    from transformers import MarianTokenizer, MarianMTModel
    model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translator = lambda x: tokenizer.batch_decode(model.generate(**tokenizer.prepare_seq2seq_batch(src_texts=x, return_tensors="pt")), skip_special_tokens=True)
    return translator

def get_moses_tokenizer(lang):
    from sacremoses import MosesTokenizer
    try:
        moses_tokenizer = MosesTokenizer(lang=lang)
    except:
        print("WARNING: Moses doesn't have tokenizer for", lang)
        moses_tokenizer = MosesTokenizer(lang=MOSES_TOKENIZER_DEFAULT_LANG)
        
    tokenizer = lambda x: moses_tokenizer.tokenize(x, return_str=True) #string IN -> string OUT
    return tokenizer

def get_bpe_segmenter(bpe_codes_path):
    from subword_nmt import apply_bpe
    bpe = apply_bpe.BPE(codes=open(bpe_codes_path, 'r'))
    segmenter = lambda x: bpe.process_line(x.strip()).split() #string IN -> list OUT
    return segmenter

def get_moses_detokenizer(lang):
    from sacremoses import  MosesDetokenizer
    try:
        moses_detokenizer = MosesDetokenizer(lang=lang)
    except:
        print("WARNING: Moses doesn't have tokenizer for", lang)
        moses_detokenizer = MosesDetokenizer(lang=MOSES_TOKENIZER_DEFAULT_LANG)
        
    tokenizer = lambda x: moses_detokenizer.detokenize(x.split(), return_str=True) #string IN -> string OUT
    return tokenizer

def tokenize_with_punkset(doc, punkset):
    tokens = []
    doc = ' '.join(doc.split())

    curr_sent = ""
    for c in doc:
        if c in punkset:
            curr_sent += c
            if curr_sent:
                tokens.append(curr_sent.strip())
                curr_sent = ""
        else:
            curr_sent += c
    if curr_sent:
        tokens.append(curr_sent.strip())            

    return tokens

def get_custom_tokenizer(punkset):
    return lambda x: tokenize_with_punkset(x, punkset)


def do_translate(model_id, text):
    if model_id in loaded_models:
        if loaded_models[model_id]['sentence_segmenter']:
            sentence_batch = loaded_models[model_id]['sentence_segmenter'](text)
        else:
            sentence_batch = [text]

        #print(sentence_batch)

        #preprocess
        #print("Preprocess")
        for step in loaded_models[model_id]['preprocessors']:
            sentence_batch = [step(s) for s in sentence_batch]
            #print(sentence_batch)

        #translate batch (ctranslate only)
        translated_sentence_batch = loaded_models[model_id]['translator'](sentence_batch)
        #print(translated_sentence_batch)

        #postprocess
        #print("Postprocess")
        tgt_sentences = translated_sentence_batch
        for step in loaded_models[model_id]['postprocessors']:
            tgt_sentences = [step(s) for s in tgt_sentences]
            #print(tgt_sentences)

        tgt_text = " ".join(tgt_sentences)

        return tgt_text
    else:
        return 0

#Functional
def read_config(config_file):
    with open(config_file, "r") as jsonfile: 
        data = json.load(jsonfile) 
        print("Config Read successful") 
        print(data)
    return data

def load_models(config_path):
    config_data = read_config(CONFIG_JSON_PATH)

    for model_config in config_data['models']:
        if model_config['load']:

            if 'alt' in model_config:
                alt_id = model_config['alt']
            else:
                alt_id = None
            
            model_id = get_model_id(model_config['src'], model_config['tgt'], alt_id)
            loaded_models[model_id] = {}
            loaded_models[model_id]['src'] = model_config['src']
            loaded_models[model_id]['tgt'] = model_config['tgt']
            
            #Load sentence segmenter
            if model_config['sentence_split'] == "nltk":
                loaded_models[model_id]['sentence_segmenter'] = nltk_sentence_segmenter
            elif type(model_config['sentence_split']) == list:
                loaded_models[model_id]['sentence_segmenter'] = get_custom_tokenizer(model_config['sentence_split'])
            else:
                loaded_models[model_id]['sentence_segmenter'] = None
            
            #Load model pipeline
            print("Model: %s ( "%model_id, end="")
            loaded_models[model_id]['preprocessors'] = []
            if 'lowercase' in model_config['pipeline'] and model_config['pipeline']['lowercase']:
                print("lowercase", end=" ")
                loaded_models[model_id]['preprocessors'].append(lowercaser)

            if 'tokenize' in model_config['pipeline'] and model_config['pipeline']['tokenize']:
                print("tokenize", end=" ")
                loaded_models[model_id]['preprocessors'].append(get_moses_tokenizer(model_config['src']))

            if 'bpe' in model_config['pipeline'] and model_config['pipeline']['bpe']:
                print("bpe", end=" ")
                model_dir = os.path.join(config_data['models_root'], model_config['model_path'])
                loaded_models[model_id]['preprocessors'].append(get_bpe_segmenter(os.path.join(model_dir, model_config['bpe_file'])))
            elif model_config['model_type'] == 'ctranslator2':
                loaded_models[model_id]['preprocessors'].append(token_segmenter)

            if 'translate' in model_config['pipeline'] and model_config['pipeline']['translate']:
                print("translate", end="")
                if model_config['model_type'] == 'ctranslator2':
                    print("-ctranslator2", end=" ")
                    model_dir = os.path.join(config_data['models_root'], model_config['model_path'])
                    loaded_models[model_id]['translator'] = get_batch_ctranslator(model_dir)  
                elif model_config['model_type'] == 'opus':
                    loaded_models[model_id]['translator'] = get_batch_opustranslator(loaded_models[model_id]['src'], loaded_models[model_id]['tgt'])
                    print("-opus-huggingface", end=" ")
            else:
                loaded_models[model_id]['translator'] = dummy_translator

            loaded_models[model_id]['postprocessors'] = []
            if 'bpe' in model_config['pipeline'] and model_config['pipeline']['bpe']:
                print("unbpe", end=" ")
                loaded_models[model_id]['postprocessors'].append(desegmenter)
            elif model_config['model_type'] == 'ctranslator2':
                loaded_models[model_id]['postprocessors'].append(token_desegmenter)

            if 'tokenize' in model_config['pipeline'] and model_config['pipeline']['tokenize']:
                print("detokenize", end=" ")
                loaded_models[model_id]['postprocessors'].append(get_moses_detokenizer(model_config['tgt']))

            if 'recase' in model_config['pipeline'] and model_config['pipeline']['recase']:
                print("recase", end=" ")
                loaded_models[model_id]['postprocessors'].append(capitalizer)

            print(")")
    
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

@translate.post('/', status_code=200)
async def translate_sentence(request: TranslationRequest):

    model_id = get_model_id(request.src, request.tgt, request.alt)

    if not model_id in loaded_models:
        raise HTTPException(status_code=401, detail="Language pair %s is not supported."%(request.src + "-" + request.tgt))
    
    translation = do_translate(model_id, request.text)

    response = TranslationResponse(translation=translation)
    return response

@translate.post('/batch', status_code=200)
async def translate_batch(request: BatchTranslationRequest):
    print(request.texts)
    print(type(request.texts))

    model_id = get_model_id(request.src, request.tgt, request.alt)

    if not model_id in loaded_models:
        raise HTTPException(status_code=401, detail="Language pair %s is not supported."%(request.src + "-" + request.tgt))
    
    translated_batch = []
    for sentence in request.texts:
        translation = do_translate(model_id, sentence)
        translated_batch.append(translation)

    response = BatchTranslationResponse(translation=translated_batch)
    return response


@translate.on_event("startup")
async def startup_event():
    load_models(CONFIG_JSON_PATH)


