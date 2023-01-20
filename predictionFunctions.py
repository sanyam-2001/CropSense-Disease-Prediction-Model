from tensorflow import keras
import cv2
import numpy as np
import json
import tensorflow as tf
from PIL import Image

modelMNv2_224 = keras.models.load_model('./models/MNv2_224.h5')
modelMNv2_256 = keras.models.load_model('./models/MNv2_256.h5')
model_Customnet = keras.models.load_model('./models/Custom.h5')

class_names = ['Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


def predictMNv2_224(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    predict = modelMNv2_224.predict(img)
    idx = np.argmax(predict)
    f = open('./categories.json')
    data = json.load(f)
    for key, value in data.items():
        if value == idx:
            ans = key
    return ans


def predictMNv2_256(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 256, 256, 3)
    predict = modelMNv2_256.predict(img)
    idx = np.argmax(predict)
    f = open('./categories.json')
    data = json.load(f)
    for key, value in data.items():
        if value == idx:
            ans = key
    return ans


def predict_custom(img):
    PIL_IMG = Image.open(img)
    PIL_IMG = PIL_IMG.resize((256, 256), resample=0)
    img_array = tf.keras.preprocessing.image.img_to_array(PIL_IMG)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model_Customnet.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class
