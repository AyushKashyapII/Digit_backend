# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image, ImageOps
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pixel-digit-painter.vercel.app"],  
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"], 
    allow_headers=["*"], 
)


model = tf.keras.models.load_model("mnist_model.h5")


class ImageRequest(BaseModel):
    image: str  

@app.post("/predict")
def predict_digit(request: ImageRequest):
    try:
        header, base64_data = request.image.split(",", 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert("L")  
        image = image.resize((28, 28), resample=Image.Resampling.LANCZOS)

        if np.mean(image) > 127:
            image = ImageOps.invert(image)

        image_array = np.array(image).astype("float32") / 255.0 
        image_array = image_array.reshape(1, 28, 28, 1)       

        prediction = model.predict(image_array)
        predicted_digit = int(np.argmax(prediction))

        return {"prediction": predicted_digit}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
