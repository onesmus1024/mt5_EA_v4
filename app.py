import os
import tensorflow as tf
import sys
import numpy as np
import pandas as pd
from flask import Flask
from flask import request, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import model


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict', methods=['POST'])
def predict():
    mark_state =[]
    #['open', 'high', 'low', 'tick_volume', 'spread', 'real_volume']
    data = request.get_json(force=True)

    mark_state.append(data['open'])
    mark_state.append(data['high'])
    mark_state.append(data['low'])
    mark_state.append(data['tick_volume'])
    mark_state.append(data['spread'])
    mark_state.append(data['real_volume'])
   
    prediction = model.predict(mark_state)
    return jsonify({'prediction': prediction})
