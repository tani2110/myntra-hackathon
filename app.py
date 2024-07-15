from flask import Flask, render_template, jsonify, request
import os
import random
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import joblib
from sklearn.preprocessing import LabelEncoder  # Add this import
from PIL import Image
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Define user preferences and dataset path
user_preferences = {'skirt': 0, 'dress': 0, 'pants': 0}
folders = ['skirt', 'pants', 'dress']  # Add this definition
test_image_path = './test_image4.jpeg'

 
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

model_path = 'clothing_classification_model.h5'
model = load_model(model_path)

encoder = LabelEncoder()
encoder.classes_ = np.array(folders)




# test_image = cv2.imread(test_image_path)
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# test_image_resized = cv2.resize(test_image, (224, 224))
 # ....
# test_image_embedding = get_embedding(test_image_path)
# test_image_embedding = np.expand_dims(test_image_embedding, axis=0)

# test_prediction = model.predict(test_image_embedding)
# predicted_class = encoder.inverse_transform(test_prediction)

# test_probabilities = model.predict_proba(test_image_embedding)

# print(f'Test image embedding shape: {test_image_embedding.shape}')
# print(f'Test prediction raw output: {test_prediction}')
# print(f'Predicted class: {predicted_class[0]}')

# for idx, class_name in enumerate(encoder.classes_):
#     print(f'Probability for {class_name}: {test_probabilities[0][idx]:.4f}')


image_urls = {
  'skirt': [
      "https://i.pinimg.com/564x/e4/cd/82/e4cd826d234f122c50f3a967a6aa7737.jpg",
      "https://i.pinimg.com/736x/c8/83/ce/c883ce3b22674188898aee38bb5c2ca2.jpg",
      "https://i.pinimg.com/736x/f0/0a/44/f00a44a959bc2ef647ad6e740c72436a.jpg",
    ],
  'pants': [
        'https://i.pinimg.com/736x/b1/9e/54/b19e54a02d815beba3ca8fec1bd1bcd0.jpg',
        'https://i.pinimg.com/736x/e2/8c/fc/e28cfc482b62c45954d5d42e8cc5045f.jpg',
       'https://i.pinimg.com/564x/e0/21/54/e02154fedbd00a448e9999aeadee6490.jpg',
    ],
  'dress': [
      'https://i.pinimg.com/736x/26/14/45/261445f8bcacffd1ee0dd69e836d9be0.jpg',
      'https://i.pinimg.com/564x/c7/08/f2/c708f2afddff3634868f634a81243a75.jpg',
    ]
}

def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    # embedding = base_model.predict(img_data)
    # return embedding[0]
    return img_array

    # Endpoint to serve the initial HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to classify the image and update user preferences
@app.route('/classify_image', methods=['POST'])
def classify_image():
    data = request.get_json()
    image_url = data.get('image_url')
    print("yes")

    # Process the image and get embedding
    response = requests.get(image_url, stream=True)
    img = Image.open(response.raw).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # return img_array

    # img_data = np.array(img)
    # img_data = np.expand_dims(img_data, axis=0)
    # img_data = preprocess_input(img_data)
    # embedding = base_model.predict(img_data)
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = encoder.classes_[predicted_class[0]]

    # predicted_class = encoder.inverse_transform(prediction)[0]
    # print(predicted_class)
    
    # Update user preferences
    user_preferences[predicted_class_name] += 1
    
    print(f"Predicted class name: {predicted_class_name}")

    return jsonify({'result':  predicted_class_name})

@app.route('/recommend_images', methods=['GET'])
def recommend_images():
    total_clicks = sum(user_preferences.values())

    if total_clicks == 0:
        return jsonify([])

    recommendations = []
    for category, clicks in user_preferences.items():
        if clicks > 0:
            proportion = clicks / total_clicks
            num_images = int(proportion * len(image_urls[category]))
            recommendations.extend(random.sample(image_urls[category], num_images))
    
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)