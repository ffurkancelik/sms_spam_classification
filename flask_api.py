from flask import Flask, request, jsonify
import dataset_helper_functions as dh
from methods import ML_Models
import os

path = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

#It is better to use a config file or args to make a dynamic look.
model_type = "transformer"
model_path = os.path.join(path, "models", model_type) #change it if you are using a transformer based vectorizer.
model_name = "KNN" #Chose your model from models dir

ml = ML_Models(model_path)
ml.load_model(model_name)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "Text field is required"}), 400
    if model_type == 'tf-idf':
        df = dh.prepare_data_for_method_ml_predict(data["text"], model_path)
        prediction = ml.predict(model_name, df)
    elif model_type == 'transformer':
        df = dh.prepare_data_for_transformer_predict(data["text"])
        prediction = ml.predict(model_name, df)
    else:
        raise FileNotFoundError(f"Model Type Not Found: {model_type}")

    prediction_label = "Spam" if prediction[0] == 1 else "Ham"
    return jsonify({"prediction": prediction_label})


if __name__ == '__main__':
    app.run(debug=True)
