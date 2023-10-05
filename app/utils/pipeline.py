import os
from typing import Dict, List, Callable, Callable

from app.exceptions import ModelLoadingException
from app.utils.segmenters import (
    desegmenter,
    get_bpe_segmenter,
    get_sentencepiece_desegmenter,
    get_sentencepiece_segmenter,
    nltk_sentence_segmenter,
    token_desegmenter,
    token_segmenter,
)
from app.utils.tokenizers import (
    get_custom_tokenizer,
    get_moses_detokenizer,
    get_moses_tokenizer,
)
from app.utils.translators import (
    get_batch_ctranslator,
    get_batch_opustranslator,
    get_batch_opusbigtranslator,
    get_batch_nllbtranslator,
    get_batch_m2m100translator,
    dummy_translator,
    get_custom_translator,
)
from app.utils.utils import (
    capitalizer,
    lowercaser,
)

from app.settings import DEFAULT_NLLB_MODEL_TYPE, DEFAULT_M2M100_MODEL_TYPE
from app.constants import NLLB_CHECKPOINT_IDS, M2M100_CHECKPOINT_IDS

def load_model_sentence_segmenter(
    model: Dict,
    model_config: Dict,
    pipeline_msg: List[str],
    *args,
    **kwargs,
) -> None:
    if 'sentence_split' in model_config:
        ss = model_config['sentence_split']
        if ss == 'nltk':
            pipeline_msg.append('sentence_split-nltk')
            model['sentence_segmenter'] = nltk_sentence_segmenter
        elif isinstance(ss, list):
            pipeline_msg.append('sentence_split-custom')
            model['sentence_segmenter'] = get_custom_tokenizer(ss)


def load_model_lowercaser(
    model: Dict,
    model_config: Dict,
    pipeline_msg: List[str],
    *args,
    **kwargs,
) -> None:
    if (
        'lowercase' in model_config['pipeline']
        and model_config['pipeline']['lowercase']
    ):
        model['preprocessors'].append(lowercaser)
        pipeline_msg.append('lowercase')


def load_model_tokenizer(
    model: Dict,
    model_config: Dict,
    pipeline_msg: List[str],
    *args,
    **kwargs,
) -> None:
    if (
        'tokenize' in model_config['pipeline']
        and model_config['pipeline']['tokenize']
    ):
        tokenizer = get_moses_tokenizer(model_config['src'])
        model['preprocessors'].append(tokenizer)
        pipeline_msg.append('mtokenize')


def load_model_segmenter(
    checks: Dict,
    model: Dict,
    model_config: Dict,
    model_dir: str,
    model_id: str,
    pipeline_msg: List[str],
    warn: Callable,
    *args,
    **kwargs,
) -> None:
    if (
        model_dir
        and 'bpe' in model_config['pipeline']
        and model_config['pipeline']['bpe']
    ):
        if not 'bpe_file' in model_config:
            warn(
                f'Failed to load bpe model for {model_id}: '
                'bpe_file not specified. Skipping load.'
            )
            raise ModelLoadingException

        model_file = os.path.join(model_dir, model_config['bpe_file'])

        if not os.path.exists(model_file):
            warn(
                f'Failed to load bpe model for {model_id}: '
                f'BPE vocabulary file not found at {model_file}. Skipping load.'
            )
            raise ModelLoadingException

        bpe_segmenter = get_bpe_segmenter(model_file)

        if not bpe_segmenter:
            warn(
                f'Failed to loading bpe model {model_file} for {model_id}. Skipping load.'
            )
            raise ModelLoadingException

        model['preprocessors'].append(bpe_segmenter)
        checks['bpe_ok'] = True
        pipeline_msg.append('bpe')
    elif (
        model_dir
        and 'sentencepiece' in model_config['pipeline']
        and model_config['pipeline']['sentencepiece']
    ):
        if not model_dir:
            warn(
                f'Failed to load sentencepiece model for {model_id}: '
                'model_path not specified. Skipping load.'
            )
            raise ModelLoadingException

        if not 'src_sentencepiece_model' in model_config:
            warn(
                f'Failed to load sentencepiece model for {model_id}: '
                'src_sentencepiece_model not specified. Skipping load.'
            )
            raise ModelLoadingException

        model_file = os.path.join(
            model_dir, model_config['src_sentencepiece_model']
        )
        if not os.path.exists(model_file):
            warn(
                f'Failed to load sentencepiece model for {model_id}: '
                f'Sentencepiece model file not found at {model_file}. Skipping load.'
            )
            raise ModelLoadingException

        model['preprocessors'].append(get_sentencepiece_segmenter(model_file))
        checks['sentencepiece_ok'] = True
        pipeline_msg.append('sentencepiece')
    elif model_config['model_type'] == 'ctranslator2':
        # default tokenizer needed for ctranslator2 translation
        model['preprocessors'].append(token_segmenter)


def load_model_translator(
    model: Dict,
    model_config: Dict,
    model_dir: str,
    model_id: str,
    pipeline_msg: List[str],
    warn: Callable,
    *args,
    **kwargs,
) -> None:
    if (
        'translate' in model_config['pipeline']
        and model_config['pipeline']['translate']
    ):
        msg = 'translate'
        if model_config['model_type'] == 'ctranslator2':
            if not model_dir:
                warn(
                    f'Failed to load ctranslate model for {model_id}: '
                    'model_path not specified. Skipping load.'
                )
                raise ModelLoadingException

            model['translator'] = get_batch_ctranslator(model_dir, 
                                                        is_multilingual=model_config.get('multilingual'), 
                                                        lang_map=model_config.get('lang_code_map'))
            msg += '-ctranslator2'
        elif model_config['model_type'] == 'opus':
            opus_translator = get_batch_opustranslator(
                model['src'], model['tgt']
            )
            if opus_translator:
                model['translator'] = opus_translator
                msg += '-opus-huggingface'
            else:
                warn(
                    f'Failed to load opus-huggingface model for {model_id}. Skipping load.'
                )
                raise ModelLoadingException
        elif model_config['model_type'] == 'opus-big':
            opus_translator = get_batch_opusbigtranslator(
                model['src'], model['tgt']
            )
            if opus_translator:
                model['translator'] = opus_translator
                msg += '-opusbig-huggingface'
            else:
                warn(
                    f'Failed to load opusbig-huggingface model for {model_id}. Skipping load.'
                )
                raise ModelLoadingException
        elif model_config['model_type'] == 'nllb':
            nllb_checkpoint_id = model_config.get('checkpoint_id') if 'checkpoint_id' in model_config else DEFAULT_NLLB_MODEL_TYPE

            if len(model_config.get('checkpoint_id').split('/')) == 1:
                if nllb_checkpoint_id not in NLLB_CHECKPOINT_IDS:
                    warn(
                        f'No checkpoint exists for base NLLB model: facebook/{nllb_checkpoint_id}. Skipping load.'
                    )
                    raise ModelLoadingException
                nllb_checkpoint_id = 'facebook/' + nllb_checkpoint_id
                warn(f'Full model id: {nllb_checkpoint_id}')

            translator = get_batch_nllbtranslator(nllb_checkpoint_id, lang_map=model_config.get('lang_code_map'))
            if translator:
                model['translator'] = translator
                msg += '-nllb-huggingface-' + nllb_checkpoint_id
            else:
                warn(
                    f'Failed to load nllb-huggingface model for {model_id}. Skipping load.'
                )
                raise ModelLoadingException
        elif model_config['model_type'] == 'm2m100':
            m2m100_checkpoint_id = model_config.get('checkpoint_id') if 'checkpoint_id' in model_config else DEFAULT_M2M100_MODEL_TYPE

            if len(model_config.get('checkpoint_id').split('/')) == 1:
                if m2m100_checkpoint_id not in M2M100_CHECKPOINT_IDS:
                    warn(
                        f'No checkpoint exists for base M2M100 model: facebook/{m2m100_checkpoint_id}. Skipping load.'
                    )
                    raise ModelLoadingException
                m2m100_checkpoint_id = 'facebook/' + m2m100_checkpoint_id
                warn(f'Full model id: {m2m100_checkpoint_id}')

            translator = get_batch_m2m100translator(m2m100_checkpoint_id, lang_map=model_config.get('lang_code_map'))
            if translator:
                model['translator'] = translator
                msg += '-m2m100-huggingface-' + m2m100_checkpoint_id
            else:
                warn(
                    f'Failed to load m2m100-huggingface model for {model_id}. Skipping load.'
                )
                raise ModelLoadingException
        elif model_config['model_type'] == 'dummy':
            msg += '-dummy'
            model['translator'] = dummy_translator
        elif model_config['model_type'] == 'custom':
            msg += '-custom'
            model['translator'] = get_custom_translator(model_config['model_path'])
        pipeline_msg.append(msg)
    else:
        model['translator'] = None


def load_model_desegmenter(
    checks: Dict,
    model: Dict,
    model_config: Dict,
    model_dir: str,
    model_id: str,
    pipeline_msg: List[str],
    warn: Callable,
    *args,
    **kwargs,
) -> None:
    if checks['bpe_ok']:
        model['postprocessors'].append(desegmenter)
        pipeline_msg.append('unbpe')
    elif checks['sentencepiece_ok']:
        if not 'tgt_sentencepiece_model' in model_config:
            warn(
                f'Failed to load sentencepiece model for {model_id}: '
                'tgt_sentencepiece_model not specified. Skipping load.'
            )
            raise ModelLoadingException

        model_file = os.path.join(
            model_dir, model_config['tgt_sentencepiece_model']
        )

        model['postprocessors'].append(
            get_sentencepiece_desegmenter(model_file)
        )
        pipeline_msg.append('desentencepiece')
    elif model_config['model_type'] == 'ctranslator2':
        model['postprocessors'].append(token_desegmenter)


def load_model_detokenizer(
    model: Dict,
    model_config: Dict,
    pipeline_msg: List[str],
    *args,
    **kwargs,
) -> None:
    if (
        'tokenize' in model_config['pipeline']
        and model_config['pipeline']['tokenize']
    ):
        detokenizer = get_moses_detokenizer(model['tgt'])
        model['postprocessors'].append(detokenizer)
        pipeline_msg.append('mdetokenize')


def load_model_recaser(
    model: Dict,
    model_config: Dict,
    pipeline_msg: List[str],
    *args,
    **kwargs,
) -> None:
    if (
        'recase' in model_config['pipeline']
        and model_config['pipeline']['recase']
    ):
        model['postprocessors'].append(capitalizer)
        pipeline_msg.append('recase')


pipeline: List[Callable] = [
    load_model_sentence_segmenter,
    load_model_lowercaser,
    load_model_tokenizer,
    load_model_segmenter,
    load_model_translator,
    load_model_desegmenter,
    load_model_detokenizer,
    load_model_recaser,
]
