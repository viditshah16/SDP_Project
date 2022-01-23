from json import load
from cv2 import CascadeClassifier
from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from video import livevideo 
from PIL import Image

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after' , methods=['GET' , 'POST'])
def after():
    # img , path = livevideo()
    # image = request.files['file1']
    # image.save('static/file.jpg')
    # img = cv2.imread('static/file.jpg' , 0)
    # image = cv2.imread(path)

    image  = livevideo()
    img = cv2.imread(image , 0)

    img = cv2.resize(img , (48,48))
    img = np.reshape(img ,(1,48,48,1))
    

    model = load_model('emotion_model.h5')

    prediction = model.predict(img)

    label_map =['Anger' , 'Neutral' , 'Fear' , 'Happy' , 'Sad' , 'Surprise']

    prediction = np.argmax(prediction)

    final = label_map[prediction]

    return render_template('after.html' , data=final , path = image)

if __name__ == "__main__":
    app.run(debug=True)
