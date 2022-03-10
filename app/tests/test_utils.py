from app.utils.utils import get_model_id, parse_model_id


def test_get_model_id():
    assert 'en-fr' == get_model_id('en', 'fr')
    assert 'en-fr-xyz' == get_model_id('en', 'fr', 'xyz')


def test_parse_model_id():
    assert parse_model_id('en') is None
    assert ('en', 'fr', '') == parse_model_id('en-fr')
    assert ('en', 'fr', 'xyz') == parse_model_id('en-fr-xyz')
