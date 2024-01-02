# working but dont know what it does

import cv2
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model
import base64


model = load_model('waste.h5')


cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((242, 208))
    img_array = np.array(img)

    prediction = model.predict(np.expand_dims(img_array, axis=0))
    print(prediction) #class_name

    # Display the result
    cv2.imshow("Result", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    #cv2.imshow("Result", img)