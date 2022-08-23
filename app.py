# Flask utils
from flask import Flask, redirect, url_for, request, render_template
import pickle
import numpy as np

# Define Flask App
app = Flask(__name__)

# -------------------------Deploying of Front Page----------------------------
@app.route('/')
def front_page():
    return render_template('heart.html')

model_heart = pickle.load(open('final_model.pkl', 'rb'))
@app.route('/heart',methods=['POST','GET'])
def heart_page():
    if request.method == 'GET':
        return render_template('heart.html')
    else:
        age = request.form['age']
        sex = request.form.get('sex')
        cp = request.form['cp'] 
        tresrbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs'] 
        restecg = request.form['restecg'] 
        thalach = request.form['thalach']
        exang = request.form['exang'] 
        oldpeak = request.form['oldpeak'] 
        slope = request.form['slope'] 
        ca = request.form['ca'] 
        thal= request.form['thal']

    input_data = (age, sex, cp, tresrbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal)
    print(input_data)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model_heart.predict(input_data_reshaped)
    senddata=""
    if (prediction[0]== 0):
        senddata='According to the given details person does not have any Heart Diesease.'
    else:
        senddata='According to the given details chances of having Heart Diesease are High, So Please Consult a Doctor'
    return render_template('heart.html', prediction_text=senddata)


if __name__ == "__main__":
    app.run(debug=True)
