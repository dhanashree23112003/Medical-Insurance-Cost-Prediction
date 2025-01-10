from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)



# Load the trained model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('prediction.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    
    # Create input array for prediction
    input_features = np.array([[age, bmi, children, smoker]])
    
    # Predict the output
    prediction = model.predict(input_features)[0]
    return render_template('prediction.html', prediction_text=f'Predicted Insurance Cost: ${prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
