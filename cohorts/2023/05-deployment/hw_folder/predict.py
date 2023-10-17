import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = './model1.bin'
dv_file = './dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dict_vectorizer = pickle.load(f_in)

app = Flask('credit_score')


@app.route('/predict', methods=['POST'])
def predict_pron():

    customer = request.get_json()

    X = dict_vectorizer.transform([customer])
    y_pred = model.predict_proba(X)[:,1]

    result = {
        'credit_score' : float(y_pred)
    }
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port =9696)


