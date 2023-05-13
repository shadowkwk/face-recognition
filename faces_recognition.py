from flask import Flask, render_template, Response, jsonify
import cv2
import time
from flask_mysqldb import MySQL
import json
import face_recognition
import os
import numpy as np
import pyttsx3

app = Flask(__name__)

path = 'images' # folder name
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # append all the image names in the array


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]  # encode images
        encodeList.append(encoded_face)
    return encodeList


encoded_face_train = findEncodings(images)  # save all trained faces

camera = cv2.VideoCapture(0)    # take pictures from webcam
retval = ""
old_time = new_time = time.time()


def gen_frames():  # generate frame by frame from camera
    while True:
        global new_time, old_time
        new_time = time.time()
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)  # scaled down the capture image by 4 times
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)    # convert the color
            faces_in_frame = face_recognition.face_locations(imgS)  # locate the faces in the capture image
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame) # encore the faces in the capture image
            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                matches = face_recognition.compare_faces(encoded_face_train, encode_face)   # compare the trained faces with the the faces in the capture image
                faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
                matchIndex = np.argmin(faceDist)    # check the most similar face
                
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper().lower()   # get the face name
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 125)
                    engine.say("welcome "+name)
                    engine.runAndWait()
                    engine.stop()
                    
                    print(name)
                    y1, x2, y2, x1 = faceloc    # top left and bottom right corner of the face
                    # since we scaled down by 4 times
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1+6, y2-5),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)  # draw the name

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route("/result", methods=['POST', 'GET'])
def result():
    global new_time, old_time
    return ""


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
