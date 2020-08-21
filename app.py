# AUTHOR: JATIN CHAUHAN

from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
from util import base64_to_pil
import IC

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app')
def application():
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

        IMAGE_PATH = "image.png"
        result = IC.function_pred_grad_cam(IMAGE_PATH)

        val1 = str(result)
        val2 = str(result)
        return jsonify(result=val1, probability=val2)

    return None

@app.route('/generate/<path>')
def generate_report(path):
    IMAGE_PATH = "2237.png"
    result = IC.function_pred_grad_cam(IMAGE_PATH)

    if result == 'DONE':
        return render_template('result.html')
    return render_template('app.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0')
