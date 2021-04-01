# write some code for the API here
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict_fare")
def predict(key,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):
    X_test = pd.DataFrame()
    X_test['key'] = [key]
    X_test['pickup_datetime']=[pickup_datetime]
    X_test['pickup_longitude'] = [pickup_longitude]
    X_test['pickup_latitude'] = [pickup_latitude]
    X_test['dropoff_longitude']=[dropoff_longitude]
    X_test['dropoff_latitude'] = [dropoff_latitude]
    X_test['passenger_count'] = [passenger_count]

    model = joblib.load('/home/rexelardo/code/rexelardo/solution_07-Data-Engineering_03-Train-at-scale_03-Train-taxiFare-on-gcp/TaxiFareModel/TaxiFareModel/model.joblib')
    prediction = model.predict(X_test)[0]

    return {'prediciton': prediction}