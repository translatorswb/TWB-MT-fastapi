import json

from fastapi import status
from fastapi.testclient import TestClient

from main import app
from app.helpers.config import Config


class TestTranslateApiV2:
    API_VERSION = 2
    SERVICE = 'translate'

    def setup(self):
        self.client = TestClient(app)
        self.config_data = {
            'languages': {},
            'models': [],
        }
        self.config = Config(config_data=self.config_data)

    def get_endpoint(self, endpoint: str = '/') -> str:
        endpoint = f'/{endpoint}' if not endpoint.startswith('/') else endpoint
        return f'/api/v{self.API_VERSION}/{self.SERVICE}{endpoint}'

    def test_async_translate_text_valid_code(self):
        options = {
            'src': 'en',
            'tgt': 'fr',
            'text': 'Hello there, how are you doing?',
        }
        response = self.client.post(
            self.get_endpoint('/'), data=json.dumps(options)
        )
        assert response.status_code == status.HTTP_200_OK
        task_content = response.json()
        assert task_content['status'] == 'SUCCESS'

    def test_async_batch_translate_text_valid_code(self):
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

        task_content = response.json()
        assert task_content['status'] == 'SUCCESS'
