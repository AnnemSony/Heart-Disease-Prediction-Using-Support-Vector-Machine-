from flask import Flask, render_template, request
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
 
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    trestbps = int(request.form.get('trestbps'))
    chol = int(request.form.get('chol'))
    fbs = int(request.form.get('fbs'))
    restecg = int(request.form.get('restecg'))
    thalach = int(request.form.get('thalach'))
    exang = int(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

 
    if sex not in [0, 1]:
        error_message = 'Invalid input for sex. Please select 0 for Female or 1 for Male.'
        return render_template('error.html', error=error_message)

   
    if cp not in [0, 1, 2, 3]:
        error_message = 'Invalid input for chest pain type. Please select a valid option.'
        return render_template('error.html', error=error_message)

  
    if restecg not in [0, 1, 2]:
        error_message = 'Invalid input for resting electrocardiographic results. Please select a valid option.'
        return render_template('error.html', error=error_message)

 
    if ca not in [0, 1, 2, 3, 4]:
        error_message = 'Invalid input for number of major vessels. Please select a valid option.'
        return render_template('error.html', error=error_message)


    if thal not in [0, 1, 2, 3]:
        error_message = 'Invalid input for thalassemia. Please select a valid option.'
        return render_template('error.html', error=error_message)

    selected_model = request.form.get('model')
    with open(f'{selected_model}_model.pkl', 'rb') as file:
        model = pickle.load(file)

   
    data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(data)

    result = f'Prediction for {selected_model}: {"Risk of Heart Disease" if prediction == 1 else "No of Risk Heart Disease"}'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
