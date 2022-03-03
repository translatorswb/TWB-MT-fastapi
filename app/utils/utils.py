from app.constants import MODEL_TAG_SEPARATOR


def lowercaser(word: str) -> str:
    return word.lower()


def capitalizer(word: str) -> str:
    return word.capitalize()


def get_model_id(src, tgt, alt_id=None):
    model_id = src + MODEL_TAG_SEPARATOR + tgt
    if alt_id:
        model_id += MODEL_TAG_SEPARATOR + alt_id
    return model_id


def parse_model_id(model_id):
    fields = model_id.split(MODEL_TAG_SEPARATOR)
    if len(fields) == 2:
        alt = ''
    elif len(fields) == 3:
        alt = fields[2]
    else:
        return False

    src = fields[0]
    tgt = fields[1]

    return src, tgt, alt
