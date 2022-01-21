from json import load
from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after' , methods=['GET' , 'POST'])
def after():
    img = request.files['file1']
    img.save('static/file.jpg')

    image = cv2.imread('static/file.jpg' , 0)
    image = cv2.resize(image , (48,48))

    image = np.reshape(image , (1,48,48,1))

    model = load_model('emotion_model.h5')
    prediction = model.predict(image)

    label_map =['Anger' , 'Neutral' , 'Fear' , 'Happy' , 'Sad' , 'Surprise']

    prediction = np.argmax(prediction)

    final = label_map[prediction]

    return render_template('after.html' , data=final)

if __name__ == "__main__":
    app.run(debug=True)
