from django.db import models
import pickle
import pandas as pd
import os
from PIL import Image
import numpy as np
import requests


def send_message(prompt):
    API_KEY = 'AIzaSyD5eAYRj9Db29f4KCzCv5CMxEh5K2GIdwY'
    API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}'

    if prompt.lower() == "exit":
        return "Gemini Chatbot: Goodbye!"
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        response_data = response.json()
        return response_data['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Error: {response.status_code} - {response.text}"


# Create your models here.

def predictDL(image_file):
    """
    Takes an uploaded image file and returns the prediction ('Cat 🐱' or 'Dog 🐶')
    """
    from tensorflow.keras.models import load_model
    MODEL_PATH = os.path.join('DL/cats_dogs_classifier.h5')  # adjust as needed
    model = load_model(MODEL_PATH)

    try:
        image = Image.open(image_file).convert('RGB')
        image = image.resize((384, 384))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)[0][0]
        return "Dog 🐶" if prediction <= 0.5 else "Cat 🐱"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unable to classify"

def predict(custom_dict):
    with open('ML/wine_model_bundle.pkl', 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    num_imputer = bundle['num_imputer']
    scaler = bundle['scaler']
    encoder = bundle['encoder']
    num_cols = bundle['num_cols']
    cat_cols = bundle['cat_cols']
    custom_input = pd.DataFrame([custom_dict])

    # Match types and process input
    for col in cat_cols:
        custom_input[col] = custom_input[col].astype(str)
    custom_input[cat_cols] = custom_input[cat_cols].fillna('Unknown')
    custom_input[num_cols] = num_imputer.transform(custom_input[num_cols])
    custom_input[num_cols] = scaler.transform(custom_input[num_cols])

    custom_encoded = encoder.transform(custom_input[cat_cols])
    custom_encoded_df = pd.DataFrame(custom_encoded, columns=encoder.get_feature_names_out(cat_cols))

    custom_final = pd.concat([
        pd.DataFrame(custom_input[num_cols], columns=num_cols),
        custom_encoded_df
    ], axis=1)

    # Prediction
    predicted_price = model.predict(custom_final)
    
    return predicted_price


import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

class Data(models.Model):
    name = models.CharField(max_length=100, null=True)
    Text = models.CharField(max_length=500,blank=False)
    Text = models.TextField(default="")

    predictions=models.CharField(max_length=100,null=True)

    def save(self, *args, **kwargs):
        rfc = joblib.load('AI/sentiment_model.joblib')
        vec = joblib.load('AI/vectorizer.joblib')
        l=[]
        l.append(self.Text)
        corpus = []
        new_item = re.sub('[^a-zA-Z]',' ',l[0])
        new_item = new_item.lower()
        new_item = new_item.split()
        lm=WordNetLemmatizer()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
        input=corpus
        transformed_input=vec.transform(input)
        if rfc.predict(transformed_input)==[0]:
            self.predictions = "Negative"
        else:
            self.predictions ="Positive"
        return super().save(*args, *kwargs)

    class Meta:
        app_label = 'WineApp'
        managed = True

    def __str__(self):
        return self.name  


