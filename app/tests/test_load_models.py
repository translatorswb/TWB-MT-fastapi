from app.helpers.config import Config

def test_load_sigle_model_with_warnings():
    config_data = {
        'languages': {
            'en': 'English',
            'fr': 'French',
        },
        'models': [
            {
                'src': 'fr',
                'tgt': 'en',
                'model_type': 'opus',
                'load': True,
                'sentence_split': 'nltk',
                'pipeline': {
                    'lowercase': True,
                    'translate': True,
                    'recase': True,
                },
            },
        ],
    }
    config = Config(config_data=config_data)
    expected_msg = 'Model: fr-en ( sentence_split-nltk lowercase translate-opus-huggingface recase )'
    assert len(config.messages) == 1
    assert config.messages[0] == expected_msg
    expected_warning = "Model path not specified for model fr-en. Can't load custom translation model or segmenters."
    assert len(config.warnings) == 1
    assert config.warnings[0] == expected_warning
