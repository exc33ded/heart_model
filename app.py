# Importing essential libraries
from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np

filename = 'final_model.pkl'
model_heart = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

# Rendering main page
@app.route('/')
def home():
	return render_template('frontpage.html')

@app.route('/heart')
def heart_page():
    return render_template('heart.html')

@app.route('/heartPredict', methods=['GET','POST'])
def Heart_predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model_heart.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
'''For Postman'''
@app.route("/predict",methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp'] 
    tresrbps = request.form['tresrbps']
    chol = request.form['chol']
    fbs = request.form['fbs'] 
    restecg = request.form['restecg'] 
    thalach = request.form['thalach']
    exang = request.form['exang'] 
    oldpeak = request.form['oldpeak'] 
    slope = request.form['slope'] 
    ca = request.form['ca'] 
    thal= request.form['thal']
    input_data = (age, sex, cp, tresrbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = int(model_heart.predict(input_data_reshaped)[0])
    data=  {
        'age': age,
        'sex': sex,
        'cp': cp,
        'tresrbps': tresrbps,
        'chol': chol,
        'fbs':fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak':oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'prediction': prediction
    }
    return jsonify(data) 

if __name__ == '__main__':
	app.run(debug=True)
