# TWB-MT-fastapi

API for serving machine translation models. 

It can run two types of models:
- [ctranslate2](https://github.com/OpenNMT/CTranslate2) models
- _Helsinki-NLP_ models provided through [huggingface](https://huggingface.co/Helsinki-NLP).

Model specifications need to go in `config.json`.

## Model installation examples

### Custom model (ctranslate2)

For an English to Turkish model, place the following files under `model/entr`: 

- ctranslator2 model as `model.bin`
- (Optional) BPE subword codes file (e.g. `bpe.en-tr.codes`)
- (Optional) Sentencepiece model (To be implemented)

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

### Helsinki-NLP models (huggingface)

You can serve Helsinki-NLP models provided in huggingface, as long as they are one-to-one. Make sure they are listed in https://huggingface.co/Helsinki-NLP and place the language codes exactly as they are. 
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

## Build and run

Set the environment variables:
```
MT_API_CONFIG=config.json
MODELS_ROOT=models
MT_API_DEVICE=gpu|cpu
```

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

## Build and run with docker-compose

### To use GPU

Do the following edits on docker-compose file
1. Remove comment `runtime: nvidia` line
2. Under environment, set MT_API_DEVICE=gpu

```
docker-compose build
docker-compose up
```

## Example calls

### Translation

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

### Retrieve languages

#### cURL

```
curl 'http://127.0.0.1:8001/api/v1/translate/languages'
```

#### Python
```
import httpx
translate_service_url = "http://127.0.0.1:8001/api/v1/translate/languages"
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

