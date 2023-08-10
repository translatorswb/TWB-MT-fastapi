# TWB-MT-fastapi

REST API for serving machine translation models in production. 

It can serve three types of translation systems:
- [ctranslate2](https://github.com/OpenNMT/CTranslate2) models
- Certain transformer-based models models provided through [huggingface](https://huggingface.co/).
    - OPUS and OPUS-big models of [Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
    - [NLLB](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/nllb) (Multilingual)
    - [M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100) (Multilingual)
- Custom e.g. rule-based translators specified as a python module (Experimental)
   
## Model configuration

Model specifications for TWB-MT-fastapi need to be provided in the `config.json` file. The configuration file enables users to specify the desired translation models, settings, pre and postprocessors they wish to use. Make sure to follow the guidelines in the documentation while configuring the models to ensure smooth integration.

### Configuration file syntax

API configuration file (`config.json`) is where we specify the models to load and their pipeline. It is a JSON format file containing a dictionary `languages` and a list `models`. Languages is just an (optional) mapping between language codes (e.g. `en`) and language names (e.g. English). `model` lists the model configurations as dictionaries. A minimal example of configuration file for serving an OPUS-big English-to-Turkish model:

```
{
  "languages": {
    "en": "English",
    "tr": "Turkish"
  },
  "models": [
    {
        "src": "en",
        "tgt": "tr",
        "model_type": "opus-big",
        "load": true,
        "sentence_split": "nltk", 
        "pipeline": {
            "translate": true
        }
    }
  ]
}
```

### Custom ctranslate2 model configuration

To load an English to Turkish model, place the following files under `<MODELS_ROOT>/entr`: 

- ctranslator2 model as `model.bin`
- (Optional) Shared BPE subword codes file `bpe_file` (e.g. `bpe.en-tr.codes`)
- (Optional) Sentencepiece model (`src_sentencepiece_model` and `tgt_sentencepiece_model`) (Example of this is shown below)

Add the following configuration under `models` in `config.json`:

```
{
    "src": "en",
    "tgt": "tr",
    "model_type": "ctranslator2",
    "model_path": "entr",  //model directory name under models
    "bpe_file": "bpe.en-tr.codes",
    "load": true,
    "sentence_split": "nltk",
    "pipeline": {
        "lowercase": true,  //make true if you processed your data in lowercase when training your model 
        "tokenize": true,  //make true if you processed your data with moses tokenizer when training your model 
        "bpe": true,  //make true if you use bpe subwording when training your model 
        "translate": true,  
        "recase": true  // true if you want to do recasing on output (capitalizes first letter of sentence)
    }
}
```

### OPUS model configuration

You can serve Helsinki-NLP's OPUS models provided in huggingface, as long as they are one-to-one. Make sure they are listed in https://huggingface.co/Helsinki-NLP and place the language codes exactly as they are. 
For an French to English model, add the following configuration under `models` in `config.json`:

```
{
    "src": "fr",
    "tgt": "en",
    "model_type": "opus",
    "load": true,
    "sentence_split": "nltk" ,
    "pipeline": {
        "lowercase": false,
        "translate": true,
        "recase": true
    }
}
```

Some models have a different architecture like the [English to Turkish model](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr). These models needs to have `opus-big` as `model_type`. For example:

```
{
    "src": "en",
    "tgt": "tr",
    "model_type": "opus-big",
    "load": true,
    "sentence_split": "nltk", 
    "pipeline": {
        "translate": true
    }
}
```

### NLLB model

[NLLB (No Language Left Behind)](https://github.com/facebookresearch/fairseq/tree/nllb) is a multilingal MT model developed by Meta AI that supports [200 languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200). Model checkpoints of various sizes are [supported through huggingface](https://huggingface.co/docs/transformers/model_doc/nllb) and can be loaded in the API by specifying the checkpoint id and language pairs to be activated in the API configuration. 

Example configuration supporting bidirectional English-Kanuri, English-French, English-Fulfulde, English-Hausa:

```
{
  "model_type": "nllb",
  "checkpoint_id": "nllb-200-distilled-600M", 
  "multilingual": true,
  "load": true,
  "sentence_split": "nltk",
  "supported_pairs": ["en-kr", "en-fr", "en-ff", "en-ha"],
  "pipeline": {
      "translate": true
  },
  "lang_code_map": {"en": "eng_Latn", "tr": "tur_Latn", "fr": "fra_Latn",
           "kr": "knc_Latn", "ha": "hau_Latn", "ff": "fuv_Latn","rw": "kin_Latn"}
}
```

Depending on your server architecture, you can choose `checkpoint_id` from `nllb-200-distilled-1.3B`, `nllb-200-distilled-600M` or `nllb-200-3.3B`.

By convention, this API uses two lettered ISO language codes in the configuration file. Since NLLB model uses a different language code convention, you need to create a mapping in the configuration for the correct use of the model (`lang_code_map`). The complete list of 200 languages and their respective codes can be viewed through [Flores200 README file](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200). (If you prefer, you can use the NLLB id directly in the configuration file as well. )

### M2M100 model

[M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100) is a multilingal MT model developed by Meta AI that supports 100 languages. Model checkpoints of various sizes ([418M](https://huggingface.co/facebook/m2m100_418M), [1.2B](https://huggingface.co/facebook/m2m100_1.2B)) are supported through huggingface and can be loaded in the API by specifying the checkpoint id and language pairs to be activated in the API configuration. 

Example configuration supporting bidirectional English-Turkish and English-Spanish:

```
{
  "model_type": "m2m100",
  "checkpoint_id": "m2m100_418M", 
  "multilingual": true,
  "load": true,
  "sentence_split": "nltk",
  "supported_pairs": ["en-tr", "en-es"],
  "pipeline": {
      "translate": true
  }
}
```

Depending on your server architecture, you can choose `checkpoint_id` from `m2m100_418M` and `m2m100_1.2B`.

## Advanced configuration features

### Alternative model loading

By default, one model can be loaded to serve a language direction. Although, if you'd like to have multiple models for a language pair or want to have multiple multilingual models, you can use the `alt` parameter in your model configuration. For example, let's load both `opus` and `opus-big` models for `en-fr` direction from huggingface:

```
{
    "src": "en",
    "tgt": "fr",
    "model_type": "opus",
    "load": true,
    "sentence_split": "nltk", 
    "pipeline": {
        "translate": true
    }
},
{
    "src": "en",
    "tgt": "fr",
    "alt": "big",
    "model_type": "opus-big",
    "load": true,
    "sentence_split": "nltk", 
    "pipeline": {
        "translate": true
    }
}
```

To use the big model while inference request, you'll need to specify an `alt` parameter as `big`. Otherwise, it'll default to the first loaded model. (Example shown later below)

**Note**: If loading two multilingual models at the same time, make sure you add `alt` labels to them. If not only the last one will be loaded. While on inference request, you don't need to specify the alt label as it will find the first model with the language direction supported. 

### Model chaining

Model chaining is useful when you want to translate in language directions which you don't have direct models for. 

Let's say you trained bidirectional Kurmanji-English models but want to serve a translator from Kurmanji to Turkish. To serve the Kurmanji to Turkish translator, you can load the Kurmanji-English model as base and place an OPUS English-Turkish model that you already have initialized to the post translator chain (`posttranslatechain`). 

```
...
{
    "src": "en",
    "tgt": "tr",
    "model_type": "opus-big",
    "load": true,
    "sentence_split": "nltk", 
    "pipeline": {
        "translate": true
    }
},
{
    "src": "kmr",
    "tgt": "tr",
    "model_type": "ctranslator2",
    "model_path": "kmr-en",
    "src_sentencepiece_model": "sp.model",
    "tgt_sentencepiece_model": "sp.model",
    "load": true,
    "sentence_split": "nltk",
    "pipeline": {
        "lowercase": true,
        "tokenize": false,
        "sentencepiece": true,
        "translate": true,
        "recase": true
    },
    "posttranslatechain": ["en-tr"]
}
...
```

Similarly, if you want to translate from Turkish to Kurmanji, you'd put the Turkish to English model in the pre-translator chain (`pretranslatechain`).

```
...
{
    "src": "tr",
    "tgt": "en",
    "model_type": "opus",
    "load": true,
    "sentence_split": "nltk",
    "pipeline": {
        "translate": true
    }
},
{
    "src": "tr",
    "tgt": "kmr",
    "model_type": "ctranslator2",
    "model_path": "en-kmr",
    "src_sentencepiece_model": "sp.model",
    "tgt_sentencepiece_model": "sp.model",
    "load": true,
    "sentence_split": "nltk",
    "pipeline": {
        "lowercase": true,
        "tokenize": false,
        "sentencepiece": true,
        "translate": true,
        "recase": true
    },
    "pretranslatechain": ["tr-en"]
}
...
```

NOTE: You can chain as many models as you want in translate chains. 

NOTE 2: It's not possible to have non-custom models as base models when chaining. 

### Custom translator packages

Custom translator packages make it possible to have a script as translator. This feature is built for implementing rule-based translators such as transliterators, text pre/post-processors. 

```
TODO: Instructions...
```

## Build and run

Set the environment variables:
```
MT_API_CONFIG=config.json
MODELS_ROOT=../translation-models
MT_API_DEVICE=cpu #or "gpu"
```

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

You can also run `run_local.sh` directly. 

## Build and run with docker-compose (recommended)

```
docker-compose build
docker-compose up
```

### To use GPU on docker

Do the following edits on docker-compose file
1. Remove comment on `runtime: nvidia` line
2. Under environment, set `MT_API_DEVICE=gpu`
3. Build and run.

## Example calls

### Simple translation

Endpoint for translating a single phrase. 

#### cURL

```
curl --location --request POST 'http://127.0.0.1:8001/api/v1/translate' \
--header 'Content-Type: application/json' \
--data-raw '{"src":"fr", "tgt":"en", "text":"c'\''est un test"}'
```

#### Python

```
import httpx
translate_service_url = "http://127.0.0.1:8001/api/v1/translate"
json_data = {'src':'fr', 'tgt':'en', 'text':"c'est un test."}
r = httpx.post(translate_service_url, json=json_data)
response = r.json()
print("Translation:", response['translation'])
```

### Using alternative models

You can specify usage of alternative models with the `alt` parameter in your requests.

#### cURL

```
curl --location --request POST 'http://127.0.0.1:8001/api/v1/translate' \
--header 'Content-Type: application/json' \
--data-raw '{"src":"en", "tgt":"fr", "alt":"big", text":"this is a test."}'
```

#### Python

```
import httpx
translate_service_url = "http://127.0.0.1:8001/api/v1/translate"
json_data = {'src':'en', 'tgt':'fr', 'alt':'big', text':"this is a test."}
r = httpx.post(translate_service_url, json=json_data)
response = r.json()
print("Translation:", response['translation'])
```

### Batch translation

Endpoint for translating a list of sentences.

#### cURL

```
curl --location --request POST 'http://127.0.0.1:8001/api/v1/translate/batch' \
--header 'Content-Type: application/json' \
--data-raw '{"src":"en", "tgt":"fr", "texts":["hello twb", "this is another sentence"]}'
```

#### Python

```
import httpx
translate_service_url = "http://127.0.0.1:8001/api/v1/translate/batch"
json_data = {'src':'fr', 'tgt':'en', 'texts':["hello twb", "this is another sentence"]}
r = httpx.post(translate_service_url, json=json_data)
response = r.json()
print("Translation:", response['translation'])
```

### Retrieve languages

Retrieves a the list of supported languages and model pairs.

#### cURL

```
curl 'http://127.0.0.1:8001/api/v1/translate/'
```

#### Python
```
import httpx
translate_service_url = "http://127.0.0.1:8001/api/v1/translate/"
r = httpx.get(translate_url)
response = r.json()
print(response)
```

#### Response description

```
{
  "languages": {
    "ar": "Levantine Arabic",
    "en": "English",
    "fr": "French",
    "swc": "Congolese Swahili",
    "ti": "Tigrinya"
  }, "models":{"fr":{"en":["fr_en"],"swc":["fr_swc"]},"swc":{"fr":["swc_fr"]},"ti":{"en":["ti_en"]},"en":{"ti":["en_ti"]},"ar":{"en":["ar_en", "ar_en_domainspecific"]}}
}
```

- `languages`: All language codes used in the system and their respective language names.
- `models`: Three level dictionary listing the available models with the structure:
```
    <source-language>
        ↳ <target-language>
            ↳ List of <model-id>s associated with the language pair
```

Note: The third level model list is for seeing different versions of the model. There is always a default model in the language pair with id `<src>_<tgt>` and there could be alternative models with model id `<src>_<tgt>_<alt-tag>`. 

For example in the setup above, there are two alternative models in the Arabic-English direction: A default model `ar_en` and a domain specific `ar_en_domainspecific` model where domainspecific is the alternative model id. For the rest of the language pairs there is only one default model. 

