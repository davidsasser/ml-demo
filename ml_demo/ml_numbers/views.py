from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
import numpy as np
import os
import base64
import tensorflow as tf
import cv2
import json

dirname = os.path.dirname(__file__)
model = tf.keras.models.load_model(os.path.join(dirname, './'))

def index(request):

    return render(request,'index.html')

def predict(request):
    if request.method == 'POST':

        base64str = request.POST.get("imgBase64", None).split(',')[1]
        # print(base64str)
        imgdata = base64.b64decode(base64str)
        filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata)

        img = cv2.imread(os.path.join(dirname, '../some_image.jpg'), 0)
        res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        #res = tf.image.resize(img, [28,28])

        res = res.reshape(28, 28, 1)

        # Making sure that the values are float so that we can get decimal points after division
        res = res.astype('float32')

        res /= 255

        pred = model.predict(res.reshape(1,28,28,1))
        my_dict = {}
        count = 0
        for i in pred:
            for j in i:
                my_dict[count] = (j*100)
                count += 1
        prediction = pred.argmax()
        return HttpResponse(json.dumps({'prediction' : str(prediction), 'confidence': my_dict}), content_type="application/json")

    # if a GET (or any other method) we'll create a blank form
    else:
        print("uh oh")

    return render(request, 'index.html', {'prediction': prediction})