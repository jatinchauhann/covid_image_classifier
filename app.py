# AUTHOR: JATIN CHAUHAN

from flask import Flask, render_template, request, url_for, flash, redirect
# import IC
from werkzeug.exceptions import abort

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app')
def application():
    return render_template('app.html')


@app.route('/generate/<path>')
def generate_report(path):
    # IMAGE_PATH = "2237.png"
    # result = IC.function_pred_grad_cam(IMAGE_PATH)
    #
    # if result == 'DONE':
    #     return render_template('result.html')
    return render_template('app.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0')
