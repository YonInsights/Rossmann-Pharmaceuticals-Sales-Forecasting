from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('sales_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    # ...preprocessing steps...
    
    # Make predictions
    predictions = model.predict(input_df)
    
    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
