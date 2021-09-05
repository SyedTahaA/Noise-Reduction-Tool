import os
import urllib.request

import cv2
import tensorflow as tf
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template("Main.html", title="main")

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully processed and displayed above')
        return render_template('Main.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    predictor = tf.keras.models.load_model("NoiseReducer300.model")
    img = cv2.imread('static/uploads/'+filename)
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, 300, 300, 3)
    prediction = predictor.predict(img)
    prediction = prediction.reshape(300, 300, 3)
    cv2.imwrite('static/uploads/'+filename, prediction)

    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
