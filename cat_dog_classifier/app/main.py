from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

from app.utils import preprocess_image


app = FastAPI()

# Load model at startup
model = load_model("models/cat_dog_model.h5")

@app.get("/")
def read_root():
    return {"status": "OK", "message": "Cat vs Dog Classifier"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        processed = preprocess_image(img)
        prediction = model.predict(processed)
        label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = float(prediction[0][0]) if label == "Dog" else 1 - float(prediction[0][0])
        return JSONResponse({"prediction": label, "confidence": round(confidence, 4)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)