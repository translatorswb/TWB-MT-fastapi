from fastapi import status
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_list_languages():
    response = client.get('/api/v1/translate')
    assert response.status_code == status.HTTP_200_OK
