from logging import exception
from flask import Flask, redirect, render_template, request, url_for, redirect
from flask_socketio import SocketIO , join_room
from db import get_user, save_user
from video import livevideo 
import cv2
from pymongo.errors import DuplicateKeyError
from keras.models import load_model
import numpy as np
from flask_login import LoginManager, current_user, login_user , login_required , logout_user 

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = "my secret key"

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method=='POST':
        username = request.form.get('username')
        password_input =request.form.get('password')
        user = get_user(username)

        if user and user.check_password(password_input):
            login_user(user)
            return redirect(url_for('home'))
        else:
            message = 'Failed Login!'

    return render_template('login.html' , message=message)


@app.route("/signup" , methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    message = ''
    if request.method=='POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password =request.form.get('password')
        try:
            save_user(username , email , password)
            return redirect(url_for('login'))
        except DuplicateKeyError:
            message="User Already Exist"
    return render_template('signup.html' , message=message)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/chat' , methods=['GET' , 'POST'])
@login_required
def chat():
    if request.method == "POST":
        username = request.form["username"]
        room = request.form["room"]

    if username and room:
        return render_template('chat.html' , username=username , room=room)
    else: 
        return redirect(url_for('home'))



# @app.route('/after' , methods=['GET' , 'POST'])
# def after():
#     image  = livevideo()
#     img = cv2.imread(image , 0)

#     img = cv2.resize(img , (48,48))
#     img = np.reshape(img ,(1,48,48,1))
    
#     model = load_model('emotion_model.h5')
#     prediction = model.predict(img)
#     label_map =['Anger' , 'Neutral' , 'Fear' , 'Happy' , 'Sad' , 'Surprise']
#     prediction = np.argmax(prediction)
#     final = label_map[prediction]
#     return render_template('after.html' , data=final)


@socketio.on('join_room')
def handle_join_room_event(data):
    app.logger.info("{} has joined the room {}".format(data['username'] , data['room']))
    join_room(data['room'])
    socketio.emit('join_room_announcement' , data)

@socketio.on('emotion')
def emotion_handle():
    image  = livevideo()
    img = cv2.imread(image , 0)

    img = cv2.resize(img , (48,48))
    img = np.reshape(img ,(1,48,48,1))
    
    model = load_model('emotion_model.h5')
    prediction = model.predict(img)
    label_map =['😡' , '😐' , '🙂' , '😎' , '🙄' , '😋']
    prediction = np.argmax(prediction)
    final = label_map[prediction]

    socketio.emit('catch_emotion' , final)



@socketio.on('send_message')
def handle_send_message_event(data):
    app.logger.info("{} has sent message to the room {} : {}".format(data['username'] , data['room'] , data['message']))
    socketio.emit('receive_message' , data , room=data['room'])


@login_manager.user_loader
def load_user(username):
    return get_user(username)

if __name__ == '__main__':
    socketio.run(app , debug=True)