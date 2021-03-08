import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from joblib import load

# To launch the app, use `uvicorn main:app --reload`
app = FastAPI()


class PostedData(BaseModel):
    token: str
    text: str


@app.get("/welcome")
def welcome():
    return {
        "Message": ("Bonjour, ceci est la beta d'un "
                    "algorithm d'analyse de sentiment"),
        "Status Code": 200}


@app.post("/sentiment")
def return_prediction(posted_data: PostedData):
    pd_dict = posted_data.dict()

    print(pd_dict['token'])
    print(TOKEN)

    if pd_dict["token"] != TOKEN:
        raise HTTPException(status_code=401, detail="Token invalide")

    clf_pipe = load(ML_MODEL)
    prediction = clf_pipe.predict([pd_dict["text"]])[0]
    prediction = "Positif" if prediction == 1 else "Négatif"

    return {"text": pd_dict["text"],
            "prediction": prediction,
            "status_code": 200}


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    ML_MODEL = os.environ.get("MODEL_PATH")
    TOKEN = os.environ.get("TOKEN")

    uvicorn.run(app, host="0.0.0.0", port=8000, 
                log_level="info")
