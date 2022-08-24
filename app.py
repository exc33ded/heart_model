# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'final_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('frontpage.html')


@app.route('/heart', methods=['GET','POST'])
def heart_page():
    if request.method == 'GET':
        return render_template('heart.html')
    else:
        age = request.form['age']
        sex = request.form['sex'] 
        cp = request.form['cp'] 
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs'] 
        restecg = request.form['restecg'] 
        thalach = request.form['thalach']
        exang = request.form['exang'] 
        oldpeak = request.form['oldpeak'] 
        slope = request.form['slope'] 
        ca = request.form['ca'] 
        thal= request.form['thal']

    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal)
    print(input_data)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    senddata=""
    if (prediction[0]== 0):
        senddata='According to the given details person does not have any Heart Diesease.'
    else:
        senddata='According to the given details chances of having Heart Diesease are High, So Please Consult a Doctor'
    return render_template('heart.html', prediction_text=senddata)
        
'''For Postman'''
@app.route("/predict",methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp'] 
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs'] 
    restecg = request.form['restecg'] 
    thalach = request.form['thalach']
    exang = request.form['exang'] 
    oldpeak = request.form['oldpeak'] 
    slope = request.form['slope'] 
    ca = request.form['ca'] 
    thal= request.form['thal']
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = int(model.predict(input_data_reshaped)[0])
    data=  {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
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
