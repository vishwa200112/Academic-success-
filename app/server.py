from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('app/model.pkl')  # Load your trained model
CORS(app)

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(file)
        
        # Ensure the data is in the correct format for your model
        # Example: you might need to preprocess or check column names
        
        # Make predictions (assuming the model expects the same features as the CSV columns)
        predictions = model.predict(data)
        
        # Convert predictions to a list for easy JSON serialization
        return jsonify({'prediction': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
