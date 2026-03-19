from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

app = Flask(__name__)
seeds_model = pickle.load(open('models/seeds_model.pkl','rb'))
crop_model = pickle.load(open('models/crop_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/seeds')
def seeds():
    return render_template('seeds.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/predict_seeds',methods = ['POST'])
def predict_seeds():
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

@app.route('/predict_crop',methods = ['POST'])
def predict_crop():
    features = [
        float(request.form['N']),
        float(request.form['P']),
        float(request.form['K']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    features_array = np.array([features])
    prediction = crop_model.predict(features_array)[0]
    return jsonify({'prediction':str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

