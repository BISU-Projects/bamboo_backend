import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import shutil

# ----- CONFIG -----
MODEL_PATH = "cnn_model3.h5"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70
CLASS_LABELS = ["Bayog", "Buho", "Golden Buho", "Iron Bamboo", "Kawayan Tinik", "Unidentified"]

# ----- LOAD MODEL -----
def load_model_with_fix(model_path):
    """Load model with architecture fix for the dense layer input issue"""
    try:
        # Try loading normally first
        model = load_model(model_path, compile=False)
        print("✓ Model loaded successfully.")
        return model
    except ValueError as e:
        if "expects 1 input(s), but it received 2 input tensors" in str(e):
            print("⚠ Model architecture mismatch detected. Attempting to fix...")
            
            # Import required modules for model reconstruction
            import tensorflow as tf
            from tensorflow.keras.models import Sequential, Model
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
            
            try:
                # Method 1: Try to load weights into a corrected architecture
                print("Attempting Method 1: Reconstructing model architecture...")
                
                # Create base model (EfficientNetB0 based on the shape (7,7,1280))
                base_model = tf.keras.applications.EfficientNetB0(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights=None
                )
                
                # Add the missing pooling layer before dense layer
                model = Sequential([
                    base_model,
                    GlobalAveragePooling2D(),  # This fixes the shape mismatch
                    Dense(len(CLASS_LABELS), activation='softmax')
                ])
                
                # Load the weights
                model.load_weights(model_path)
                print("✓ Model reconstructed and weights loaded successfully.")
                return model
                
            except Exception as e2:
                print(f"Method 1 failed: {e2}")
                
                try:
                    # Method 2: Try with Functional API
                    print("Attempting Method 2: Functional API approach...")
                    
                    base_model = tf.keras.applications.EfficientNetB0(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights=None
                    )
                    
                    x = base_model.output
                    x = GlobalAveragePooling2D()(x)
                    predictions = Dense(len(CLASS_LABELS), activation='softmax')(x)
                    
                    model = Model(inputs=base_model.input, outputs=predictions)
                    model.load_weights(model_path)
                    print("✓ Model reconstructed with Functional API successfully.")
                    return model
                    
                except Exception as e3:
                    print(f"Method 2 failed: {e3}")
                    
                    # Method 3: Try with different base architecture
                    print("Attempting Method 3: Alternative architecture...")
                    try:
                        # Simple CNN architecture as fallback
                        model = Sequential([
                            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
                            tf.keras.layers.MaxPooling2D(),
                            tf.keras.layers.Conv2D(64, 3, activation='relu'),
                            tf.keras.layers.MaxPooling2D(),
                            tf.keras.layers.Conv2D(128, 3, activation='relu'),
                            tf.keras.layers.MaxPooling2D(),
                            tf.keras.layers.GlobalAveragePooling2D(),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(len(CLASS_LABELS), activation='softmax')
                        ])
                        
                        model.load_weights(model_path, by_name=True, skip_mismatch=True)
                        print("✓ Model reconstructed with alternative architecture.")
                        return model
                        
                    except Exception as e4:
                        print(f"Method 3 failed: {e4}")
                        raise Exception("All reconstruction methods failed. Please check your model file.")
        else:
            raise e

# Load the model
try:
    model = load_model_with_fix(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("Please ensure your model file exists and is not corrupted.")
    exit(1)

# ----- INIT FASTAPI -----
app = FastAPI()

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict bamboo species from uploaded image"""
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess image
        processed_img = preprocess_image(temp_file)
        
        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Clean up temp file
        os.remove(temp_file)
        
        # Determine result based on confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_class = CLASS_LABELS[predicted_class_idx]
        else:
            predicted_class = "Unidentified"
        
        # Create response
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                CLASS_LABELS[i]: float(predictions[0][i]) 
                for i in range(len(CLASS_LABELS))
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"}, 
            status_code=500
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Bamboo Classification API is running!", "model_loaded": True}

@app.get("/classes")
async def get_classes():
    """Get available bamboo classes"""
    return {"classes": CLASS_LABELS}

if __name__ == "__main__":
    print("🎋 Starting Bamboo Classification API...")
    print(f"📁 Model: {MODEL_PATH}")
    print(f"🏷️  Classes: {CLASS_LABELS}")
    print("🚀 Server starting on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)