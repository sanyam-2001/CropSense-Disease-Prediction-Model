from tensorflow import keras
import cv2
import numpy as np
import json


def predictMNv2_224(path):
    model = keras.models.load_model('./models/MNv2_224.h5')
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    predict = model.predict(img)
    idx = np.argmax(predict)
    f = open('./categories.json')
    data = json.load(f)
    for key, value in data.items():
        if value == idx:
            ans = key
    return ans


def predictMNv2_256(path):
    model = keras.models.load_model('./models/MNv2_256.h5')
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 256, 256, 3)
    predict = model.predict(img)
    idx = np.argmax(predict)
    f = open('./categories.json')
    data = json.load(f)
    for key, value in data.items():
        if value == idx:
            ans = key
    return ans


def predictPool(path):
    pathX = './files/' + path
    predictionModels = [predictMNv2_224, predictMNv2_256]
    ans = []
    for model in predictionModels:
        ans.append(model(pathX))
    return ans
