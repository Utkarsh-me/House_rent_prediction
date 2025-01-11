from flask import Flask
from flask import render_template
import pickle
import numpy as np
from flask import request

model = pickle.load(open('house_rent.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('rent_land.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    try:
        property_type = (request.form['property_type'])
        bhk = (request.form['bhk'])
        size = (request.form['size'])
        floor = (request.form['floor'])
        area_type = (request.form['area_type'])
        state = (request.form['state'])
        furnishing_status = (request.form['furnishing_status'])
        tenant_preferred = (request.form['tenant_preferred'])
        bathrooms = (request.form['bathrooms'])
        point_of_contact = (request.form['point_of_contact'])
        
        # Prepare input data for the model
        arr = np.array([[property_type, bhk, size, floor, area_type, state, furnishing_status, tenant_preferred, bathrooms, point_of_contact]])
        prediction = model.predict(arr)
        

        return render_template('rent_res.html', prediction = prediction)
    except Exception as e:
        return render_template('rent_res.html', error=str(e))


if __name__=="__main__":
    app.run(debug=True)

