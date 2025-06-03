
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

letters_model = load_model("emnist_letters_model.keras")
byclass_model = load_model("emnist_byclass_model.keras")

letters_label_map = {i: chr(65 + i) for i in range(26)}
byclass_label_map = {i: chr(65 + i) for i in range(26)}

@app.post("/predict_letters/")
async def predict_letters(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype("float32") / 255.0
    prediction = letters_model.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    character = letters_label_map[predicted_class]
    return JSONResponse(content={"predicted_class": predicted_class, "character": character})

@app.post("/predict_byclass/")
async def predict_byclass(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype("float32") / 255.0
    prediction = byclass_model.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    character = byclass_label_map[predicted_class]
    return JSONResponse(content={"predicted_class": predicted_class, "character": character})
