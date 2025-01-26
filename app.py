from flask import Flask, request, jsonify, render_template
from supabase import create_client, Client
import pickle
import numpy as np
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Supabase credentials
SUPABASE_URL = "https://ivarjhpbeasabwwtmvnx.supabase.co"  # Add your Supabase URL here
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml2YXJqaHBiZWFzYWJ3d3Rtdm54Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzc4MjA3NDksImV4cCI6MjA1MzM5Njc0OX0.ZxUCDNIPBWLjJuN71fc7xlTdplOqNGRI2JPPpEjfW9U"  # Add your Supabase Service Role Key here

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load ML model
model = pickle.load(open('house_rent.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('rent_land.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    try :
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


            data = {
                'property_type': property_type,
                'bhk': bhk,
                'size': size,
                'floor': floor,
                'area_type': area_type,
                'state': state,
                'furnishing_status': furnishing_status,
                'tenant_preferred': tenant_preferred,
                'bathrooms': bathrooms,
                'point_of_contact': point_of_contact,
                'predicted_rent': float(prediction[0])
            }
            response = supabase.table("rent_prediction").insert(data).execute()

        if response.status_code != 201:
            return f"An error occurred while saving to Supabase: {response.json()}"

        return render_template('rent_res.html', prediction=prediction)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
