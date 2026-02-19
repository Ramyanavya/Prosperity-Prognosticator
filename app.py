from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = [float(request.form[f]) for f in features]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "SUCCESS - High chance of Startup Success!" if prediction == 1 else "FAILED - Low chance of Startup Success."
        return render_template('result.html', prediction=result)
    return render_template('predict.html', features=features)

if __name__ == '__main__':
    app.run(debug=True)
