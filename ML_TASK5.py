import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to recognize food items and estimate calorie content from an image
def recognize_food(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    food_items = []
    for _, food, confidence in decoded_predictions:
        food_items.append((food, confidence))

    return food_items

# Example usage
image_path = 'path/to/your/image.jpg'
food_items = recognize_food(image_path)
for food, confidence in food_items:
    print(food, confidence)