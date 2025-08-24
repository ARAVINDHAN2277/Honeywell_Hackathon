import requests
from datetime import datetime, timedelta
import json


# Mumbai Airport coordinates
lat, lon = 19.0896, 72.8656

# Get min/max date from flight.xlsx
import pandas as pd
flight_df = pd.read_excel('flight.xlsx')
min_date = pd.to_datetime(flight_df['Date'], errors='coerce').min().strftime('%Y-%m-%d')
max_date = pd.to_datetime(flight_df['Date'], errors='coerce').max().strftime('%Y-%m-%d')
start_date = min_date
end_date = max_date

url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,cloudcover,windspeed_10m&start_date={start_date}&end_date={end_date}&timezone=Asia%2FKolkata"
response = requests.get(url)
if response.status_code == 200:
    weather = response.json()
    print("Weather data fetched.")
else:
    print(f"Failed to fetch weather data: {response.status_code}")
    weather = {}

with open('weather_data.json', 'w') as f:
    json.dump(weather, f, indent=2)
print("Weather data saved to weather_data.json")
