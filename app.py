# AUTHOR: JATIN CHAUHAN

import os
from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from util import base64_to_pil
from werkzeug.utils import secure_filename
import IC


UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
FILENAMEINPUT = ''


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'classifier'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            FILENAMEINPUT = 'input.' + filename.rsplit('.', 1)[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], FILENAMEINPUT))

    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app', methods=['GET', 'POST'])
def application():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            FILENAMEINPUT = 'input.' + filename.rsplit('.', 1)[1]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], FILENAMEINPUT))
        # ADD THE CODE HERE TO RUN THE MODEL

        IMAGE_PATH = FILENAMEINPUT
        result = ""
        result = IC.function_pred_grad_cam(IMAGE_PATH)

        return render_template('appoutput.html')


    return render_template('app.html')


@app.route('/appbeta')
def applicationbeta():
    return render_template('appbeta.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("image.png")

        # IMAGE_PATH = "FILENAMEINPUT"
        result = ""
        # result = IC.function_pred_grad_cam(IMAGE_PATH)

        val1 = str(result)
        val2 = str(result)
        return jsonify(result=val1, probability=val2)

    return None


if __name__ == '__main__':
    #TODO Remove the debug statement before puching it to production
    app.run(host='0.0.0.0')
