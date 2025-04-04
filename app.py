from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load scaler and label encoders
with open(os.path.join(os.path.dirname(__file__), "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

# Load model metrics
with open(os.path.join(os.path.dirname(__file__), "model_metrics.pkl"), "rb") as f:
    model_metrics = pickle.load(f)

# Available models with standardized names
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "XGBoost": "xgboost.pkl"
}

# Mapping frontend model names to match keys in model_metrics.pkl
model_name_map = {
    "Logistic Regression": "LogisticRegression",
    "Decision Tree": "DecisionTree",
    "Random Forest": "RandomForest",
    "SVM": "SVM",
    "Gradient Boosting": "GradientBoosting",
    "XGBoost": "XGBoost"
}

def load_model(model_name):
    """Loads the selected model from pickle files."""
    with open(os.path.join(os.path.dirname(__file__), model_files[model_name]), "rb") as f:
        return pickle.load(f)

@app.route('/')
def index():
    """Renders the index page with the model selection dropdown."""
    return render_template('index.html', models=model_files.keys())

@app.route('/get_metrics', methods=['POST'])
def get_metrics():
    """Returns the performance metrics of the selected model."""
    data = request.json
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Model not selected"}), 400
    
    model_key = model_name_map.get(model_name)  # Standardize model name
    metrics = {key: round(value, 4) for key, value in model_metrics.get(model_key, {}).items()}
    return jsonify({"metrics": metrics})

@app.route('/predict_page', methods=['GET'])
def predict_page():
    """Redirects to the prediction page with the selected model."""
    model_name = request.args.get("model")
    if not model_name:
        return redirect(url_for("index"))  # Redirects if no model is selected
    
    return render_template('predict.html', model=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request with the selected model."""
    data = request.json
    model_name = data.get("model")
    features = data.get("features")

    if not model_name or not features:
        return jsonify({"error": "Model or features missing"}), 400

    features = np.array(features).reshape(1, -1)
    
    # Scale input features
    features = scaler.transform(features)
    
    # Load selected model and make a prediction
    model = load_model(model_name)
    prediction = model.predict(features)[0]
    
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
