import json

from fastapi import status

from app.tests.base_test_case import APIBaseTestCase


class TestTranslateApiV1(APIBaseTestCase):
    API_VERSION = 1

    def test_list_languages(self):
        response = self.client.get(url=self.get_endpoint('/'))
        assert response.status_code == status.HTTP_200_OK

        content = response.json()
        assert content['models'] == self.config.languages_list
        assert content['languages'] == self.config.language_codes

    def test_translate_text_valid_code(self):
        options = {
            'src': 'en',
            'tgt': 'fr',
            'text': 'Hello there, how are you doing?',
        }
        expected_translation = 'Bonjour, comment allez-vous?'
        response = self.client.post(
            self.get_endpoint('/'), data=json.dumps(options)
        )
        assert response.status_code == status.HTTP_200_OK
        content = response.json()
        assert content['translation'] == expected_translation

    def test_translate_text_invalid_code(self):
        options = {
            'src': 'en',
            'tgt': 'xyz',
            'text': 'Hello there, how are you doing?',
        }
        response = self.client.post(
            self.get_endpoint('/'), data=json.dumps(options)
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_batch_translate_text_valid_code(self):
        options = {
            'src': 'en',
            'tgt': 'fr',
            'texts': ['Hello, what is your name?', 'How are you doing?'],
        }
        expected_translations = [
            'Bonjour, quel est votre nom?',
            'Comment ça va?',
        ]
        response = self.client.post(
            url=self.get_endpoint('/batch'), data=json.dumps(options)
        )
        assert response.status_code == status.HTTP_200_OK

        content = response.json()
        assert content['translation'] == expected_translations

    def test_batch_translate_text_invalid_code(self):
        options = {
            'src': 'en',
            'tgt': 'xyz',
            'texts': ['Hello, what is your name?', 'How are you doing?'],
        }
        response = self.client.post(
            url=self.get_endpoint('/batch'), data=json.dumps(options)
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
