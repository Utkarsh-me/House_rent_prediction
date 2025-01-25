from flask import Flask
from flask import render_template
import pickle
import numpy as np
from flask import request
import firebase_admin
from firebase_admin import credentials, db
import requests
import os
import json

#FIREBASE_KEY_URL = "house-rent-prediction-43bd8-firebase-adminsdk-fbsvc-bec842c544.json"
#response = requests.get(FIREBASE_KEY_URL)
#if response.status_code == 200:
#    key_data = response.json()
#else:
#    raise ValueError("Failed to load Firebase key from URL")
    
#cred = credentials.Certificate(key_data)

#___________________________________________

#FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY")
#if not FIREBASE_KEY_PATH:
#    raise ValueError("FIREBASE_KEY_PATH environment variable not set")
#cred = credentials.Certificate(FIREBASE_KEY_PATH)

#___________________________________________

key_url = os.environ.get('FIREBASE_KEY')
cred = credentials.Certificate(key_url)

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://house-rent-prediction-43bd8-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

model = pickle.load(open('house_rent.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('rent_land.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
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


        try:
            ref = db.reference('predictions')
            ref.push({
                'property_type' : property_type,
                'bhk' : bhk,
                'size' : size,
                'floor' : floor,
                'area_type' : area_type,
                'state' : state,
                'furnishing_status' : furnishing_status,
                'tenant_preferred' : tenant_preferred,
                'bathrooms' : bathrooms,
                'point_of_contact': point_of_contact,
                'predicted_rent' : float(prediction[0])
            })

            #return jsonify({'message': 'Data saved successfully!'}), 200

            # Redirect or display a success message
            return render_template('rent_res.html', prediction = prediction)

        except Exception as e:
            return f"An error occurred: {e}"
        
    return render_template('rent_land.html')
        



if __name__=="__main__":
    app.run(debug=True)

