from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

def read_imagefile(file) -> Image.Image:
    image_data = file.read()
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
