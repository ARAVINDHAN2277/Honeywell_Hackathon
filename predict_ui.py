import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import importlib.util
import importlib.util
# Import cascading impact detection function
spec_cascade = importlib.util.spec_from_file_location("cascading_utils", "cascading_utils.py")
cascading_utils = importlib.util.module_from_spec(spec_cascade)
spec_cascade.loader.exec_module(cascading_utils)
import math
def get_live_congestion_bounding_box(lamin, lomin, lamax, lomax):
    params = {
        "lamin": lamin,
        "lomin": lomin,
        "lamax": lamax,
        "lomax": lomax
    }
    try:
        resp = requests.get("https://opensky-network.org/api/states/all", params=params, timeout=30)
        data = resp.json()
        if 'states' in data and data['states']:
            return len(data['states'])
        else:
            return 0
    except Exception as e:
        return 0

# Define bounding boxes for airports (add more as needed)
airport_bboxes = {
    'BOM': (18.0, 72.5, 19.5, 73.5),  # Mumbai
    'DEL': (28.4, 76.8, 28.8, 77.4),  # Delhi
}
import time
def get_slot_traffic_opensky(airport_icao, sched_dep_time):
    """
    Fetch slot traffic (number of departures) from OpenSky API for the given airport and 15-min window around sched_dep_time.
    Returns: int (number of flights)
    """
    try:
        # Convert sched_dep_time to UNIX timestamp
        slot_start = int(sched_dep_time.timestamp())
        slot_end = int((sched_dep_time + pd.Timedelta(minutes=15)).timestamp())
        url = f"https://opensky-network.org/api/flights/departure?airport={airport_icao}&begin={slot_start}&end={slot_end}"
        resp = requests.get(url)
        if resp.status_code == 200:
            flights = resp.json()
            return len(flights) + 1  # +1 to include the user's flight
        else:
            return 1
    except Exception as e:
        return 1
# Import slot traffic calculation function
spec = importlib.util.spec_from_file_location("slot_traffic_utils", "slot_traffic_utils.py")
slot_traffic_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slot_traffic_utils)
# Upload planned schedule
st.sidebar.header('Upload Planned Schedule (Excel)')
schedule_file = st.sidebar.file_uploader('Choose a schedule Excel file', type=['xlsx'])
if schedule_file:
    schedule_df = pd.read_excel(schedule_file)
    if 'sched_dep' not in schedule_df.columns:
        # Try to create sched_dep from Date and STD
        if 'Date' in schedule_df.columns and 'STD' in schedule_df.columns:
            schedule_df['Date'] = pd.to_datetime(schedule_df['Date'], errors='coerce')
            def combine_date_time(row):
                try:
                    t = pd.to_datetime(str(row['STD'])).time()
                    return pd.to_datetime(row['Date']).replace(hour=t.hour, minute=t.minute, second=t.second)
                except:
                    return pd.NaT
            schedule_df['sched_dep'] = schedule_df.apply(combine_date_time, axis=1)
        else:
            st.sidebar.warning('Schedule file must have sched_dep or Date and STD columns.')
else:
    schedule_df = None
import matplotlib.pyplot as plt
import requests
from datetime import datetime

st.title('Flight Delay Prediction')
# Cascading Impact Detection UI
st.header('Cascading Impact Detection')
if schedule_df is not None and st.button('Detect Cascading Impact'):
    try:
        G, critical_flights = cascading_utils.build_cascading_graph(schedule_df)
        st.subheader('Top 10 Critical Flights (by disruption risk)')
        # Show flight details for top 10
        top_df = schedule_df.loc[[f[0] for f in critical_flights]]
        st.dataframe(top_df)
        st.write('Graph summary:')
        st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        st.write('Each edge represents a tight turnaround that may propagate delay.')
    except Exception as e:
        st.error(f"Error in cascading impact detection: {e}")

# Input fields
date = st.date_input('Flight Date')
std = st.text_input('Scheduled Departure Time (e.g., 06:00 or 8:14 AM)')
atd = st.text_input('Actual Departure Time (e.g., 06:10 or 8:20 AM)')
sta = st.text_input('Scheduled Arrival Time (e.g., 08:00 or 10:00 AM)')
from_airport = st.text_input('From (e.g., Mumbai (BOM))')
to_airport = st.text_input('To (e.g., Delhi (DEL))')
aircraft = st.text_input('Aircraft Type (e.g., A320)')

# Helper functions for feature engineering
def combine_date_time(date_val, time_val):
    if not date_val or not time_val:
        return pd.NaT
    try:
        if 'AM' in time_val or 'PM' in time_val:
            t = pd.to_datetime(time_val).time()
        else:
            t = pd.to_datetime(str(time_val)).time()
        dt = pd.to_datetime(date_val).replace(hour=t.hour, minute=t.minute, second=t.second)
        return dt
    except Exception:
        return pd.NaT

def get_weather_features(dep_time):
    try:
        with open('weather_data.json', 'r') as f:
            weather = json.load(f)
        weather_hours = weather['hourly']['time']
        weather_df = pd.DataFrame({
            'weather_time': pd.to_datetime(weather_hours),
            'temperature_2m': weather['hourly'].get('temperature_2m', [np.nan]*len(weather_hours)),
            'windspeed_10m': weather['hourly'].get('windspeed_10m', [np.nan]*len(weather_hours)),
            'precipitation': weather['hourly'].get('precipitation', [np.nan]*len(weather_hours)),
            'cloudcover': weather['hourly'].get('cloudcover', [np.nan]*len(weather_hours)),
        })
        if pd.isna(dep_time):
            return [np.nan, np.nan, np.nan, np.nan]
        idx = np.argmin(np.abs((weather_df['weather_time'] - dep_time).dt.total_seconds()))
        row = weather_df.iloc[idx]
        return [row['temperature_2m'], row['windspeed_10m'], row['precipitation'], row['cloudcover']]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]

if st.button('Predict Delay'):
    # Show weather map for Mumbai (BOM) for the selected date
    try:
        # Mumbai coordinates
        lat, lon = 19.0896, 72.8656
        date_str = date.strftime('%Y-%m-%d')
        today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
        if date_str <= today_str:
            # Use archive API for past or today
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,precipitation,cloudcover&timezone=Asia%2FKolkata"
        else:
            # Use forecast API for future
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,precipitation,cloudcover&timezone=Asia%2FKolkata"
        resp = requests.get(url)
        if resp.status_code == 200:
            weather = resp.json()
            hours = pd.to_datetime(weather['hourly']['time'])
            temp = weather['hourly']['temperature_2m']
            precip = weather['hourly']['precipitation']
            cloud = weather['hourly']['cloudcover']
            fig, ax1 = plt.subplots(figsize=(8,3))
            ax1.plot(hours, temp, label='Temperature (°C)', color='tab:red')
            ax1.set_ylabel('Temperature (°C)', color='tab:red')
            ax2 = ax1.twinx()
            ax2.plot(hours, precip, label='Precipitation (mm)', color='tab:blue', linestyle='dashed')
            ax2.set_ylabel('Precipitation (mm)', color='tab:blue')
            ax1.set_xlabel('Hour')
            plt.title(f'Weather for Mumbai (BOM) on {date_str}')
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write('Could not fetch weather map for the selected date.')
    except Exception as e:
        st.write('Error displaying weather map:', str(e))
    # Feature engineering
    sched_dep = combine_date_time(date, std)
    actual_dep = combine_date_time(date, atd)
    sched_arr = combine_date_time(date, sta)

    dep_delay = (actual_dep - sched_dep).total_seconds() / 60 if pd.notna(actual_dep) and pd.notna(sched_dep) else 0
    hour = sched_dep.hour if pd.notna(sched_dep) else np.nan
    dayofweek = sched_dep.dayofweek if pd.notna(sched_dep) else np.nan
    route = f"{from_airport}-{to_airport}"
    aircraft_type = aircraft
    dep_slot_15min = sched_dep.floor('15min') if pd.notna(sched_dep) else pd.NaT
    # Calculate slot traffic for origin and destination
    congestion_details = ""
    origin_congestion = None
    dest_congestion = None
    now = pd.Timestamp.now(tz='UTC')
    # Use planned schedule for future, live data for now/near-future (within 1 hour)
    # Origin
    if schedule_df is not None and pd.notna(sched_dep):
        slot_traffic = slot_traffic_utils.calculate_slot_traffic(schedule_df, from_airport, sched_dep)
        congestion_details += f"Origin slot traffic (from schedule): {slot_traffic} flights in this 15-min window.\n"
        origin_congestion = slot_traffic
    elif from_airport.upper() in airport_bboxes and pd.notna(sched_dep):
        if abs((sched_dep.tz_localize('Asia/Kolkata').tz_convert('UTC') - now).total_seconds()) < 3600:
            bbox = airport_bboxes[from_airport.upper()]
            slot_traffic = get_live_congestion_bounding_box(*bbox)
            congestion_details += f"Origin slot traffic (live from OpenSky): {slot_traffic} flights in this area.\n"
            origin_congestion = slot_traffic
        else:
            slot_traffic = 1
            congestion_details += "Origin slot traffic: 1 (future, no schedule uploaded)\n"
            origin_congestion = slot_traffic
    else:
        slot_traffic = 1
        congestion_details += "Origin slot traffic: 1 (default, no schedule or live data)\n"
        origin_congestion = slot_traffic

    # Destination
    dest_slot_traffic = None
    if to_airport.upper() in airport_bboxes and pd.notna(sched_arr):
        if abs((sched_arr.tz_localize('Asia/Kolkata').tz_convert('UTC') - now).total_seconds()) < 3600:
            bbox = airport_bboxes[to_airport.upper()]
            dest_slot_traffic = get_live_congestion_bounding_box(*bbox)
            congestion_details += f"Destination slot traffic (live from OpenSky): {dest_slot_traffic} flights in this area.\n"
            dest_congestion = dest_slot_traffic
        else:
            congestion_details += "Destination slot traffic: 1 (future, no schedule uploaded)\n"
            dest_congestion = 1
    else:
        congestion_details += "Destination slot traffic: (no live data available)\n"

    # Show congestion details in a container
    with st.container():
        st.info(congestion_details)
        # Warn if origin or destination congestion is high
        if origin_congestion is not None and origin_congestion > 10:
            st.warning("Warning: Origin runway/slot congestion is high! Flight may not be schedulable in this window.")
        if dest_congestion is not None and dest_congestion > 10:
            st.warning("Warning: Destination runway/slot congestion is high! Flight may not be schedulable in this window.")
    temperature_2m, windspeed_10m, precipitation, cloudcover = get_weather_features(sched_dep)

    # Build DataFrame for prediction
    input_dict = {
        'hour': [hour],
        'dayofweek': [dayofweek],
        'route': [route],
        'aircraft_type': [aircraft_type],
        'slot_traffic': [slot_traffic],
        'temperature_2m': [temperature_2m],
        'windspeed_10m': [windspeed_10m],
        'precipitation': [precipitation],
        'cloudcover': [cloudcover],
        'dep_delay': [dep_delay],
    }
    input_df = pd.DataFrame(input_dict)
    # One-hot encoding to match training
    X_pred = pd.get_dummies(input_df)

    # Load model (assumes you have saved it as rf_model.pkl)
    try:
        with open('ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('ensemble_model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        for col in model_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[model_columns]
        prob = model.predict_proba(X_pred)[0,1]
        threshold = 0.2  # Lowered threshold for more sensitive delay prediction
        pred = int(prob > threshold)
        st.write(f"Predicted probability of delay: {prob:.2f}")
        st.write(f"Prediction (threshold {threshold}): {'Delayed' if pred else 'On Time'}")
    except Exception as e:
        st.write('Model or columns file not found. Please train and save your model as ensemble_model.pkl and ensemble_model_columns.pkl.')
        st.write(str(e))
