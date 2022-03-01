from app.helpers.config import Config

config = Config()

def translate_text(model_id, text):
    # print("do_translate")
    # print(text)

    if model_id in config.loaded_models:
        if config.loaded_models[model_id]['sentence_segmenter']:
            sentence_batch = config.loaded_models[model_id]['sentence_segmenter'](text)
        else:
            sentence_batch = [text]

        # print("sentence_batch")
        # print(sentence_batch)

        #preprocess
        # print("Preprocess")
        for step in config.loaded_models[model_id]['preprocessors']:
            sentence_batch = [step(s) for s in sentence_batch]
            # print(step)
            # print(sentence_batch)

        #translate batch (ctranslate only)
        if config.loaded_models[model_id]['translator']:
            translated_sentence_batch = config.loaded_models[model_id]['translator'](sentence_batch)
            # print("translated_sentence_batch")
            # print(translated_sentence_batch)
        else:
            translated_sentence_batch = sentence_batch

        #postprocess
        # print("Postprocess")
        tgt_sentences = translated_sentence_batch
        for step in config.loaded_models[model_id]['postprocessors']:
            tgt_sentences = [step(s) for s in tgt_sentences]
            # print(step)
            # print(tgt_sentences)

        tgt_text = " ".join(tgt_sentences)
        # print("tgt_text")
        # print(tgt_text)

        return tgt_text
    else:
        return 0
