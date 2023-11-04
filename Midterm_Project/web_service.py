import xgboost as xgb
import joblib
from flask import Flask 
from flask import request
from flask import jsonify


# load the model
model_file = 'model.model'
model = xgb.Booster({'nthread':4})
model.load_model(model_file)

# load the dv
dv_input_file = "dv.pkl"
dv = joblib.load(dv_input_file)

app = Flask('hear_disease_pred')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    X_test = dv.transform(customer)
    features = list(dv.get_feature_names_out())
    dtest = xgb.DMatrix(X_test, feature_names=features)
   
    y_pred = model.predict(dtest)
    disease = y_pred >= 0.5
    
    result = {
        'disease_probability': float(y_pred),
        'get_disease': bool(disease)
    }
    
    return jsonify(result) 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)