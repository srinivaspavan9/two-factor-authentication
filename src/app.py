from flask import Flask, render_template, Response, request
import cv2
import time
import os
import subprocess

#face recognition libraries
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#for face recognition
sp_image = face_recognition.load_image_file("images/srinivas.jpg")
sp_face_encoding = face_recognition.face_encodings(sp_image)[0]
sriram_image = face_recognition.load_image_file("images/sriram.jpg")
sriram_face_encoding = face_recognition.face_encodings(sriram_image)[0]
known_face_encodings = [
    sp_face_encoding,
    sriram_face_encoding
]
known_face_names = [
    "Srinivas",
    "Sri ram"
]
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cap_image',methods=['GET','POST'])
def cap_image():
    flag=True
    while(flag):
        sucs,frame=camera.read()
        if not sucs:
            return render_template('status2.html')
        else:
            cv2.imwrite('static/random2.jpg',frame)
            flag=False
    #face recognition code starts here
    unknown_image = face_recognition.load_image_file("static/random2.jpg")
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    del draw
    pil_image.save('static/random2.jpg')
    # face recognition code ends here
    return render_template('status.html')

@app.route('/finger',methods=['GET','POST'])
def finger():
    status=subprocess.call(['python3','/home/srinivaspavan/Desktop/finger_print/src/fingerprint-recognition.py'])
    return render_template('status2.html')
    # return render_template('status2.html')

@app.route('/base')
def base():
    return render_template('base.html')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
