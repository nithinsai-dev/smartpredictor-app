from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
from pyexpat import features

app = Flask(__name__)

seeds_model = pickle.load(open('models/seeds_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features = [
        float(request.form['A']),
        float(request.form['P']),
        float(request.form['C']),
        float(request.form['LK']),
        float(request.form['WK']),
        float(request.form['A_Coef']),
        float(request.form['LKG'])
    ]

    features_array = np.array([features])
    prediction = seeds_model.predict(features_array)[0]

    varieties = {0:'Kama',1:'Rosa',2:'Canadian'}
    result = varieties[prediction]

    return jsonify({'prediction':result})

if __name__ == '__main__':
    app.run(debug=True)

