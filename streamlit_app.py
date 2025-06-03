import streamlit as st
import pandas as pd
import joblib
import eda  # Ensure eda.py is in the same folder

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Flight_Price_Cleaned.csv")

    # Duration_Minutes
    df['Duration_Minutes'] = (
        df['Duration'].str.extract(r'(\d+)h').fillna(0).astype(int) * 60 +
        df['Duration'].str.extract(r'(\d+)m').fillna(0).astype(int)
    )

    # Total_Stops_Num
    stop_map = {
        'non-stop': 0, '1 stop': 1, '2 stops': 2,
        '3 stops': 3, '4 stops': 4
    }
    df['Total_Stops_Num'] = df['Total_Stops'].map(stop_map)

    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Flight Price Prediction App")
page = st.sidebar.radio("Select a page", ["Home", "EDA", "Price Prediction"])

# Page routing
if page == "Home":
    st.title("Welcome to the Flight Price Predictor")
    st.write("Use the sidebar to explore EDA or predict flight prices.")

elif page == "EDA":
    eda.display_eda(df)

elif page == "Price Prediction":
    st.title("📈 Predict Flight Price")

    # Load model
    try:
        model = joblib.load("best_model.pkl")
        model_features = model.feature_names_in_
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        st.stop()

    # User inputs
    airline = st.selectbox("Airline", df['Airline'].unique())
    source = st.selectbox("Source", df['Source'].unique())
    destination = st.selectbox("Destination", df['Destination'].unique())
    total_stops = st.selectbox("Total Stops", df['Total_Stops'].unique())
    duration = st.text_input("Duration (e.g., 2h 50m)", "2h 50m")
    dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23, value=9)
    arrival_hour = st.number_input("Arrival Hour", min_value=0, max_value=23, value=11)
    day = st.number_input("Day of Journey", min_value=1, max_value=31, value=15)
    month = st.number_input("Month of Journey", min_value=1, max_value=12, value=5)

    # Convert duration to minutes
    try:
        hours = int(duration.split('h')[0])
        minutes = int(duration.split(' ')[1].replace('m', '')) if 'm' in duration else 0
        duration_minutes = hours * 60 + minutes
    except:
        st.error("Invalid duration format. Use format like 2h 50m")
        st.stop()

    # Map total stops
    stop_map = {
        'non-stop': 0, '1 stop': 1, '2 stops': 2,
        '3 stops': 3, '4 stops': 4
    }
    total_stops_num = stop_map.get(total_stops, 1)

    # Build input dictionary
    input_dict = {
        'Duration_Minutes': duration_minutes,
        'Total_Stops_Num': total_stops_num,
        'Journey_Day': day,
        'Journey_Month': month,
        'Dep_Hour': dep_hour,
        'Arrival_Hour': arrival_hour,
    }

    # Add dummy variables for one-hot encoded features
    for col in model_features:
        if col.startswith("Airline_"):
            input_dict[col] = 1 if col == f"Airline_{airline}" else 0
        elif col.startswith("Source_"):
            input_dict[col] = 1 if col == f"Source_{source}" else 0
        elif col.startswith("Destination_"):
            input_dict[col] = 1 if col == f"Destination_{destination}" else 0
        elif col not in input_dict:
            input_dict[col] = 0  # Fill any missing columns

    # Final DataFrame
    input_df = pd.DataFrame([input_dict])[model_features]

    # Predict
    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"✈️ Predicted Flight Price: ₹{int(prediction):,}")

