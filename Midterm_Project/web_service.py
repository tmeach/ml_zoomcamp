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
    client = request.get_json()
     
    X = dv.transform([client])
    dval = xgb.DMatrix(X,feature_names=dv.get_feature_names_out())

    y_pred = model.predict(dval)
    disease = y_pred >= 0.5
        
    
    result = {
        'disease_probability': float(y_pred),
        'get_disease': bool(disease)
    }
    
    return jsonify(result) 

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)