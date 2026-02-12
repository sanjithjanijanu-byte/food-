import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Food Waste Optimizer", layout="wide")

st.title("🍽 AI-Based Food Waste Minimization Platform")

# -----------------------------
# DEMO TRAINING DATA (Synthetic)
# -----------------------------

np.random.seed(42)
data_size = 200

demo_data = pd.DataFrame({
    "DayOfWeek": np.random.randint(0,7,data_size),
    "Weather": np.random.randint(0,4,data_size),
    "Festival": np.random.randint(0,2,data_size),
    "Customers": np.random.randint(50,300,data_size),
})

demo_data["Demand"] = (
    demo_data["Customers"] * 0.8 +
    demo_data["Weather"] * 10 +
    demo_data["Festival"] * 30 +
    np.random.randint(-20,20,data_size)
)

X = demo_data[["DayOfWeek","Weather","Festival","Customers"]]
y = demo_data["Demand"]

model = RandomForestRegressor()
model.fit(X,y)

# -----------------------------
# NGO DATA
# -----------------------------

ngo_locations = {
    'No Food Waste Chennai': ('T. Nagar', 13.0427, 80.2593),
    'Chennai Food Bank': ('Anna Salai', 13.0493, 80.2596),
    'Robin Hood Army': ('Chennai Center', 13.0827, 80.2707)
}

capacities = {
    'No Food Waste Chennai': 200,
    'Chennai Food Bank': 150,
    'Robin Hood Army': 100
}

hotel_lat = 13.0827
hotel_lon = 80.2707

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# -----------------------------
# MANAGER FORM
# -----------------------------

st.header("📋 Hotel Manager Dashboard")

with st.form("manager_form"):

    day = st.selectbox("Select Day of Week",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    
    weather = st.selectbox("Weather Today",
        ["Sunny","Rainy","Cold","Hot"])
    
    festival = st.radio("Is Today Festival?", ["No","Yes"])
    
    customers = st.number_input("Expected Customers", min_value=0, value=100)
    
    prepared = st.number_input("Meals Prepared", min_value=0, value=120)
    
    submit = st.form_submit_button("Run AI Analysis")

# -----------------------------
# PROCESSING
# -----------------------------

if submit:

    day_map = {
        "Monday":0,"Tuesday":1,"Wednesday":2,
        "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6
    }

    weather_map = {
        "Sunny":0,"Rainy":1,"Cold":2,"Hot":3
    }

    festival_val = 1 if festival == "Yes" else 0

    input_data = np.array([[ 
        day_map[day],
        weather_map[weather],
        festival_val,
        customers
    ]])

    predicted_demand = model.predict(input_data)[0]

    st.subheader("📊 AI Demand Prediction")
    st.success(f"Predicted Demand: {int(predicted_demand)} meals")

    surplus = prepared - predicted_demand

    # -----------------------------
    # SURPLUS CHECK
    # -----------------------------

    if surplus > 0:

        st.error(f"⚠ Surplus Detected: {int(surplus)} meals")

        wasted_food_kg = surplus * 0.3
        meals_possible = int(wasted_food_kg / 0.3)

        st.subheader("🍱 Redistribution Analysis")
        st.write(f"Wasted food can feed **{meals_possible} people**")

        # -----------------------------
        # FIND NEAREST NGO
        # -----------------------------

        distances = {}

        for ngo, (location, lat, lon) in ngo_locations.items():
            distance = haversine(hotel_lat, hotel_lon, lat, lon)
            distances[ngo] = distance

        nearest_ngo = min(distances, key=distances.get)

        st.subheader("🚚 NGO Recommendation")
        st.success(f"Nearest NGO: {nearest_ngo}")
        st.write(f"Distance: {round(distances[nearest_ngo],2)} km")
        st.write(f"NGO Capacity: {capacities[nearest_ngo]} meals/day")

        # -----------------------------
        # WEATHER FOOD RECOMMENDATION
        # -----------------------------

        st.subheader("🌦 Weather-Based Food Recommendation")

        if weather == "Rainy":
            st.write("Soup, Tea, Pakora, Hot Noodles")
        elif weather == "Sunny":
            st.write("Juice, Ice Cream, Buttermilk")
        elif weather == "Cold":
            st.write("Coffee, Hot Meals, Stew")
        elif weather == "Hot":
            st.write("Lemonade, Salads, Cold Drinks")

    else:
        st.success("✅ No Surplus. Production Optimized.")
