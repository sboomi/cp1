import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastapi.testclient import TestClient

from .run_fastapi import app

client = TestClient(app)
load_dotenv(find_dotenv())

ROOT_DIR = Path().absolute().parents[1]
ML_MODEL = ROOT_DIR / Path(os.environ.get("MODEL_PATH"))
TOKEN = os.environ.get("TOKEN")


def test_welcome():
    response = client.get("/welcome")
    assert response.status_code == 200
    assert response.json() == {
        "Message": ("Bonjour, ceci est la beta d'un "
                    "algorithm d'analyse de sentiment")}


def test_return_prediction_good_token():
    assert ML_MODEL.exists()
    response = client.post("/sentiment/", json={
        "token": TOKEN,
        "text": "C'est un super resto!"
    })
    assert response.status_code == 200
    rep_list = list(response.json().keys())
    assert sorted(rep_list) == sorted(["text", "prediction", "status_code"])


def test_return_prediction_bad_token():
    response = client.post("/sentiment/", json={
        "token": "ABCDE",
        "text": "C'est un super resto!"
    })
    assert response.status_code == 401
