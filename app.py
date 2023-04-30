from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained decision tree model
model = joblib.load('decision_tree_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert the data into a NumPy array
        feature_array = np.array([data['features']])

        # Make a prediction using the model
        prediction = model.predict(feature_array)

        # Return the prediction result
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)