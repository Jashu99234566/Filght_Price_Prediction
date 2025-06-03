import streamlit as st
import pandas as pd
import pickle
import eda

# === Load Models and Data ===
@st.cache_data
def load_models_and_data():
    best_model = pickle.load(open("best_model.pkl", "rb"))
    flight_data = pd.read_csv(r"C:\Users\OMEN\OneDrive\Desktop\Processed_Flight_Data.csv")
    satisfaction_data = pd.read_csv(r"C:\Users\OMEN\OneDrive\Desktop\Cleaned_Passenger_Satisfaction_TreeModels.csv")
    return best_model, flight_data, satisfaction_data

model, flight_df, satisfaction_df = load_models_and_data()

# === Sidebar ===
st.sidebar.title("✈️ Unified ML App")
page = st.sidebar.radio("Go to", ["Home", "Flight Price Prediction", "Customer Satisfaction"])

# === Home ===
if page == "Home":
    st.title("📊 Welcome to Unified ML App")
    st.markdown("""
    This app integrates two projects:
    - **Flight Price Prediction** using Regression models
    - **Customer Satisfaction Prediction** using Classification models

    Use the sidebar to navigate and interact with predictions and EDA.
    """)

# === Flight Price Prediction ===
elif page == "Flight Price Prediction":
    st.title("🛫 Flight Price Prediction")
    sub = st.radio("Choose View", ["EDA", "Predict Price"])

    if sub == "EDA":
        eda.display_eda(flight_df)

    elif sub == "Predict Price":
        st.header("Predict Flight Price")
        with st.form("flight_form"):
            dep_hour = st.slider("Departure Hour", 0, 23, 9)
            arr_hour = st.slider("Arrival Hour", 0, 23, 12)
            duration_mins = st.slider("Duration (minutes)", 0, 1000, 180)
            stops = st.selectbox("Total Stops", [0, 1, 2, 3])
            submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame({
                "Dep_Hour": [dep_hour],
                "Arrival_Hour": [arr_hour],
                "Duration_Minutes": [duration_mins],
                "Total_Stops_Num": [stops],
            })
            price = model.predict(input_df)[0]
            st.success(f"Estimated Flight Price: ₹{price:.2f}")

# === Customer Satisfaction ===
elif page == "Customer Satisfaction":
    st.title("🧍 Customer Satisfaction Prediction")
    sub2 = st.radio("Choose View", ["EDA", "Predict Satisfaction"])

    if sub2 == "EDA":
        st.subheader("EDA for Passenger Data")
        st.dataframe(satisfaction_df.head())

    elif sub2 == "Predict Satisfaction":
        st.header("Predict Customer Satisfaction")
        with st.form("satisfaction_form"):
            age = st.slider("Age", 18, 85, 30)
            flight_dist = st.slider("Flight Distance", 100, 5000, 1000)
            inflight_wifi = st.slider("Inflight Wifi Service (0-5)", 0, 5, 3)
            seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 3)
            onboard_service = st.slider("Onboard Service (0-5)", 0, 5, 3)
            submit2 = st.form_submit_button("Predict")

        if submit2:
            input2 = pd.DataFrame({
                "Age": [age],
                "Flight Distance": [flight_dist],
                "Inflight wifi service": [inflight_wifi],
                "Seat comfort": [seat_comfort],
                "On-board service": [onboard_service],
            })
            result = model.predict(input2)[0]
            status = "Satisfied" if result == 1 else "Not Satisfied"
            st.success(f"Prediction: {status}")
