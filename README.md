# TWB-MT-fastapi

REST API for serving machine translation models in production. 

It can serve three types of translation systems:
- [ctranslate2](https://github.com/OpenNMT/CTranslate2) models
- Certain transformer-based models models provided through [huggingface](https://huggingface.co/).
    - OPUS and OPUS-big models of [Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
    - [NLLB](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/nllb) (Multilingual)
    - [M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100) (Multilingual)
- Custom e.g. rule-based translators specified as a python module (beta)

Features:
- Minimal REST API interface for using many models
- Batch translation
- Low-code specification for loading various types of models at start-up
- Automatic downloading of huggingface models
- Multilingual model support
- Model chaining
- Translation pipeline specification (lowercasing, tokenization, subwording, recasing)
- Automatic sentence splitting (uses `nltk` library)
- Manual sentence splitting with punctuation
- Supports sentencepiece and byte-pair-encoding models
- GPU support
- Easy deployment with docker compose 

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

### CTranslate2 model configuration (unidirectional)

CTranslate is a library for accelerating neural machine translation models, and it supports models in the `model.bin` format. To load your model, place the following files under `<MODELS_ROOT>/<model-identifier>`: 

- ctranslator2 model as `model.bin`
- (Optional) Shared BPE subword codes file `bpe_file` (e.g. `bpe.en-tr.codes`)
- (Optional) Sentencepiece model(s) (e.g. `sentencepiece.model`)

For our English-Turkish model, our entry in `config.json` would be:

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

Note that in order to enable subword segmentation in the pipeline, you need to include either `"bpe": true` or `"sentencepiece": true` in the `pipeline` variable.



### Multilingual CTranslate2 model configuration

It is also possible to load multilingual models with CTranslate2. Here's an example configuration for loading [multilingual NLLB model](https://forum.opennmt.net/t/nllb-200-with-ctranslate2/5090).

```
{
      "model_type": "ctranslator2",
      "model_path": "nllb-200-distilled-1.3B-int8",
      "src_sentencepiece_model": "flores200_sacrebleu_tokenizer_spm.model",
      "tgt_sentencepiece_model": "flores200_sacrebleu_tokenizer_spm.model",
      "multilingual": true,
      "load": true,
      "sentence_split": "nltk",
      "supported_pairs": ["en-rw", "rw-en"],
      "pipeline": {
        "lowercase": true,
        "tokenize": false,
        "sentencepiece": true,
        "translate": true,
        "recase": true
      },
      "lang_code_map": {"en": "eng_Latn", "tr": "tur_Latn", 
                        "fr": "fra_Latn", "rw": "kin_Latn"}
    }
``` 

Some things to note here: 
- We placed the `multilingual` flag to signal to the API that our model accepts language flags during inference. 
- We specified the language directions we want to serve through our API with the `supported_pairs` variable. These will be the language codes available through your API. If you make a request not in this list, the API will respond saying language pair not supported. 
- By convention, this API uses two lettered ISO language codes in the configuration file. Since NLLB model uses a different language code convention, we created a mapping in `lang_code_map` variable. The complete list of 200 languages supported by the NLLB model and their respective codes can be viewed through [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200). If you don't mind using the original language id's you could skip this and put language codes in this format directly to `supported_pairs` e.g. ["eng_Latn-kin_Latn", "kin_Latn-eng_Latn"].

### OPUS model through HuggingFace

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

### NLLB model through HuggingFace

[NLLB (No Language Left Behind)](https://github.com/facebookresearch/fairseq/tree/nllb) is a multilingal MT model developed by Meta AI that supports [200 languages](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200). Model checkpoints of various sizes are [supported through HuggingFace](https://huggingface.co/docs/transformers/model_doc/nllb) and can be loaded in the API by specifying the checkpoint id and language pairs to be activated in the API configuration. 

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

**Note**: When loading two multilingual models at the same time, you _must_ use `alt` labels. If you don't, only the last one will be loaded. Unless you have two models supporting in the same language direction, you don't need to specify the alt label in your request, as it will automatically find the  model which supports that language direction. 

### Model chaining

Model chaining is useful when you want to translate in language directions which you don't have direct models for. 

Let's say you trained bidirectional Kurmanji-English models but want to serve a translator from Kurmanji to Turkish. To serve the Kurmanji to Turkish translator, you can load the Kurmanji-English model as base and place an OPUS English-Turkish model that you  have already initialized to the post translator chain (`posttranslatechain`). 

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

**Note:** You can chain as many models as you want in translate chains. 

**Note 2:** It's only possible to have ctranslate type models as base models when chaining (for now). 

### Sentence splitting

Machine translation models are usually trained to input and output sentences. When translating a long text, it should be segmented into sentences before inputted to the MT. MT-API can perform this automatically on languages that use latin punctuation (`.`, `?`, `!` etc.) using `nltk` library. To do this you just need to add `"sentence_split": "nltk"` to the model configuration. 

If your language has different type of punctuation, then you can manually specify those punctuation marks which mark the ending of a sentence in a list. To illustrate with Tigrinya, a language that uses [Ge'ez script](https://en.wikipedia.org/wiki/Ge%CA%BDez_script), you would add this to its model configuration:

```
"sentence_split": ["፧", "።", "፨", "?", "!", ":", "“", "”", "\"", "—", "-"]
```

Note that if you don't specify a technique for sentence splitting, the whole text input in the request will be sent to the model. Long text input can overload the model or only a portion would be processed. 

### Custom translator packages (beta)

Custom translator packages make it possible to have a built-in python script as translator. This feature is built for implementing rule-based translators such as transliterators, text pre/post-processors. 

Custom translator scripts are placed under the directory `customtranslators`. To create a custom translator for a language, open a directory under `customtranslators` with the name of your language and then place the script `interface.py` under `src` there. This script needs to contain a function with the name `translate` which takes the input string and outputs its "translation".

Let's illustrate with a [Arabic chat alphabet (Arabizi)](https://en.wikipedia.org/wiki/Arabic_chat_alphabet) transliterator.  

Content of `customtranslators/arabizi/src/interface.py`:

```
arabizi_dict = {'ض': 'D', 'ص': 'S', 'ث': 'th', 'ق': 'q', 'ف': 'f', 'غ': 'gh', 'ع': '3', 'ه': 'h', 'خ': 'kh', 'ح': '7', 'ج': 'j', 'ة': 'a', 'ش': 'sh', 'س': 's', 'ي': 'ii', 'ب': 'b', 'ل': 'l', 'ا': 'aa', 'ت': 't', 'ن': 'n', 'م': 'm', 'ك': 'k', 'ظ': 'DH', 'ط': 'T', 'ذ': 'dh', 'د': 'd', 'ز': 'z', 'ر': 'r', 'و': 'uu', '،': ',', 'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ء': '2', 'أ': '2'} 

def translate(strin):
    return ''.join(arabizi_dict[c] if c in arabizi_dict else c for c in strin)

```

Entry in configuration file:

```
{
    "src": "ar",
    "tgt": "arb",
    "model_type": "custom",
    "model_path": "arabizi",
    "load": true,
    "pipeline": {
        "translate": true
    }
}
```

You can also have multiple interface files for a language like `arabizi/src/interface_to_arabizi.py`, `arabizi/src/interface_from_arabizi.py` and specify the name of the interface script file in the configuration file:

```
{
    "src": "ar",
    "tgt": "arb",
    "model_type": "custom",
    "model_path": "arabizi/interface_to_arabizi.py",
    "load": true,
    "pipeline": {
        "translate": true
    }
},
{
    "src": "arb",
    "tgt": "ar",
    "model_type": "custom",
    "model_path": "arabizi/interface_from_arabizi.py",
    "load": true,
    "pipeline": {
        "translate": true
    }
}
```

## Build and run

To run locally, you can set up a virtual environment with Python 3.8. 

Set the environment variables (linux):
```
MT_API_CONFIG=config.json
MODELS_ROOT=../translation-models #wherever you want your models to be stored
MT_API_DEVICE=cpu #or "gpu"
```

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

You can also run `run_local.sh` directly on linux. 

## Build and run with docker-compose (recommended)

```
docker-compose build
docker-compose up
```

### To use GPU on docker

Do the following edits on docker-compose file:

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

