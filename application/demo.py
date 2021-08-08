import io
import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

sys.path.append(os.environ['PWD'])

import utils  # noqa: E402

app = Flask(__name__)
CORS(app)

config = utils.load_yaml('application/core/config.yaml')
predictor = utils.create_instance(config['cifar_10'])


@app.route('/cifar10', methods=['GET'])
def template():
    return render_template('home.html')


@app.route('/cifar10', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'IMAGE Not Uploaded'

    # get file from the request
    file = request.files['file']

    # convert that file to bytes
    image_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes), mode='r').convert('RGB')
    except IOError:
        return jsonify(predictions='Image Not Found, Please Upload File Again!')

    outputs = predictor(images=[np.asarray(image)[:, :, ::-1].copy()])

    # return jsonify(predictions=outputs)

    prediction = f"{outputs[0][0]} - {outputs[0][1] * 100:.2f}%"

    return render_template("home.html", prediction=prediction)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
