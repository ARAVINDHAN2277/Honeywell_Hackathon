
# Fetch live flights over Mumbai using OpenSky public API (no authentication required)
import requests
import json

# Mumbai bounding box
lamin, lomin = 18.8, 72.6   # min latitude, min longitude
lamax, lomax = 19.3, 73.1   # max latitude, max longitude

url = f"https://opensky-network.org/api/states/all?lamin={lamin}&lomin={lomin}&lamax={lamax}&lomax={lomax}"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(f"Fetched {len(data.get('states', []))} live flights over Mumbai area.")
else:
    print(f"Failed to fetch data: {response.status_code}")
    data = {}

with open('opensky_live_mumbai.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Live flight data saved to opensky_live_mumbai.json")
