# pylint: disable=too-many-arguments
import pandas as pd
from flask import Flask, request,jsonify
import joblib   
from flask_cors import CORS
import warnings 
warnings.filterwarnings("ignore")

app=Flask(__name__)
CORS(app)

# loading the saved model from local drive
model=joblib.load('model_jbl.model')

@app.route('/predictUnnamed/<float:v_1>/<int:v_2>/<int:v_3>/<int:v_4>/<int:v_5>/<int:v_6>',methods = ['GET','POST'])
def predict_unnamed(v_1,v_2,v_3,v_4,v_5,v_6):
    """This function gets the unnamed args from the api and predicts the label"""
    prediction = model.predict([[v_1,v_2,v_3,v_4,v_5,v_6]])
    return jsonify({'Prediction': int(prediction) })


if __name__ == '__main__':
    app.run(debug=True)
