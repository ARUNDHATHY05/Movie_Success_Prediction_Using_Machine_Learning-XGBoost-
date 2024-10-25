from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved XGBoost model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Welcome page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Prediction form page
@app.route('/predict_form')
def predict_form():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    director_name = request.form['director_name']
    actor_1_name = request.form['actor_1_name']
    actor_2_name = request.form['actor_2_name']
    actor_3_name = request.form['actor_3_name']
    gross = request.form['gross']
    imdb_score = float(request.form['imdb_score'])  # Only use IMDb score for prediction

    # Create feature array using only IMDb score for prediction
    input_features = np.array([imdb_score]).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(input_features)
    
    # Prediction result (convert to label)
    prediction_label = 'flop' if prediction == 0 else 'good' if prediction == 1 else 'hit'

    return render_template('result.html', 
                           director_name=director_name, 
                           actor_1_name=actor_1_name, 
                           actor_2_name=actor_2_name, 
                           actor_3_name=actor_3_name,
                           gross=gross,
                           imdb_score=imdb_score,
                           prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
