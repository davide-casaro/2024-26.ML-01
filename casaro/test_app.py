import pytest
from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_infer(client):
    json_data = {
        "release_date": 2020.0101,
        "publisher": "Ubisoft",  
        "median_playtime": 30,
        "price": 19.99,
        "Genre: Action": 1,
        "Genre: Adventure": 0,
        "Genre: Casual": 0,
        "Genre: Early Access": 0,
        "Genre: Free to Play": 0,
        "Genre: Indie": 1,
        "Genre: Massively Multiplayer": 0,
        "Genre: RPG": 0,
        "Genre: Racing": 0,
        "Genre: Simulation": 0,
        "Genre: Sports": 0,
        "Genre: Strategy": 0
    }

    response = client.post("/infer", json=json_data)
    assert response.status_code == 200
    data = response.get_json()
    assert "value" in data["result"]
