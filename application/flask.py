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

config = utils.load_yaml('modules/application/core/config.yaml')
predictor = utils.create_instance(config['cifar_10'])


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html', value='Image')

    elif request.method == 'POST':
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

        return jsonify(predictions=outputs)


if __name__ == '__main__':
    app.run(debug=True)
