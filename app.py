from flask import Flask
from flask import render_template
import pickle
import numpy as np
from flask import request
import firebase_admin
from firebase_admin import credentials, db

# URL of the JSON file on GitHub (raw URL)
GITHUB_JSON_URL = "https://raw.githubusercontent.com/Utkarsh-me/House_rent_prediction/c933504e061b4e62b4b555ee3416926d23ed8e72/house-rent-prediction-43bd8-firebase-adminsdk-fbsvc-8d9d89150d.json"

# Fetch the JSON file
response = requests.get(GITHUB_JSON_URL)
if response.status_code == 200:
    # Save the file locally
    with open("firebase_key.json", "wb") as f:
        f.write(response.content)

    # Initialize Firebase Admin SDK
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://house-rent-prediction-43bd8-default-rtdb.asia-southeast1.firebasedatabase.app/'})

else:
    raise Exception(f"Failed to fetch JSON file: {response.status_code}")


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

