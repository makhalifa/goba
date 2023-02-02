import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile
from pydantic import BaseModel
import pickle
import tensorflow as tf
import cv2
import numpy as np

from cancer.lung import Lung

app  =  FastAPI()

host = "127.0.0.1"
port = 8000

with open("cancer/cancer.pkl", "rb") as f:
    Cancer_model = pickle.load(f)

pneumonia_model = tf.keras.models.load_model("pneumonia/model.h5")
Tuberculosis_model = tf.keras.models.load_model("tuberculosis/tuberculosis.h5")

@app.get( "/" )
def  read_root():
    return  { "Cancer" :  "POST /Cancer/predict",
            "Pneumonia" : "POST /Pneumonia/test",
            "Tuberculosis" : "POST /Tuberculosis/test"
            } 

@app.post( "/Cancer/predict" )
def predict_cancer(data: Lung):
    data = data.dict()

    Age = data["Age"]
    Gender = data["Gender"]
    AirPollution = data["AirPollution"]
    Alcoholuse = data["Alcoholuse"]
    DustAllergy = data["DustAllergy"]
    OccuPationalHazards = data["OccuPationalHazards"]
    GeneticRisk = data["GeneticRisk"]
    chronicLungDisease = data["chronicLungDisease"]
    BalancedDiet = data["BalancedDiet"]
    Obesity = data["Obesity"]
    Smoking = data["Smoking"]
    PassiveSmoker = data["PassiveSmoker"]
    ChestPain = data["ChestPain"]
    CoughingofBlood = data["CoughingofBlood"]
    Fatigue = data["Fatigue"]
    WeightLoss = data["WeightLoss"]
    ShortnessofBreath = data["ShortnessofBreath"]
    Wheezing = data["Wheezing"]
    SwallowingDifficulty = data["SwallowingDifficulty"]
    ClubbingofFingerNail = data["ClubbingofFingerNail"]
    FrequentCold = data["FrequentCold"]
    DryCough = data["DryCough"]
    Snoring = data["Snoring"]

    prediction = Cancer_model.predict([[
        Age,
        Gender,
        AirPollution,
        Alcoholuse,
        DustAllergy,
        OccuPationalHazards,
        GeneticRisk,
        chronicLungDisease,
        BalancedDiet,
        Obesity,
        Smoking,
        PassiveSmoker,
        ChestPain,
        CoughingofBlood,
        Fatigue,
        WeightLoss,
        ShortnessofBreath,
        Wheezing,
        SwallowingDifficulty,
        ClubbingofFingerNail,
        FrequentCold,
        DryCough,
        Snoring]])

    return  { "prediction" :  int(prediction[0]) }

@app.post( "/pneumonia/predict" )
async def predict_pneumonia(image: bytes = File(...)):
    # expected shape=(None, 64, 64, 3)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [64, 64])
    image = np.array(image)
    image = image.reshape(1, 64, 64, 3)
    image = image / 255.0
    image = image.astype(np.float32)
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)

    prediction = pneumonia_model.predict(image)
    prediction = prediction[0][0]
    return {"prediction": f"{prediction}"}

@app.post( "/tuberculosis/predict" )
async def predict_tuberculosis(image: bytes = File(...)):
    # image preprocessing
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [300, 300])
    image = np.array(image)
    image = image.reshape(1, 300, 300, 3)
    image = image / 255.0
    image = image.astype(np.float32)
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    
    prediction = Tuberculosis_model.predict(image)
    prediction = int(prediction[0][0])
    return {"prediction":f"{prediction}"}

if __name__ ==  "__main__" :
    uvicorn.run( "app:app" , host  , port , reload=True)