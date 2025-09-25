from app import app
import pytest

@pytest.fixture
def clients():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(clients):
    response = clients.get('/')
    assert response.status_code == 200
    assert b'Hello, World!' in response.data

def test_iris(clients):
    response = clients.post('/iris', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert  response.status_code == 200
    assert b'{"prediction":"setosa"}' in response.data

    response = clients.post('/iris', json={'features': [5.9, 3.0, 5.1, 1.8]})
    assert  response.status_code == 200
    assert b'{"prediction":"virginica"}' in response.data
    

    

    

