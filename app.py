import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import shutil

# ==== CONFIG ====
MODEL_PATH = "cnn_model3.h5"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70
CLASS_LABELS = ["Bayog", "Bolo", "Golden Buho", "Iron Bamboo", "Kawayan Tinik", "Unidentified"]

# ==== LOAD MODEL ====
model = load_model(MODEL_PATH, compile=False)
print(" Model loaded successfully.")

# ==== INIT FASTAPI ====
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess image
        img = image.load_img(temp_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Remove temp file
        os.remove(temp_path)

        # Check bamboo detection
        if confidence < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                "detected": False,
                "message": "Unidentified.",
                "confidence": float(confidence)
            })

        return JSONResponse(content={
            "detected": True,
            "class": CLASS_LABELS[predicted_class_index],
            "confidence": float(confidence)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
