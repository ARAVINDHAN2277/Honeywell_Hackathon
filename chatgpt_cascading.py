import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from datetime import datetime, timedelta, timezone
import requests

# --- Import optimizer ---
spec_opt = importlib.util.spec_from_file_location("schedule_optimizer", "schedule_optimizer.py")
schedule_optimizer = importlib.util.module_from_spec(spec_opt)
spec_opt.loader.exec_module(schedule_optimizer)

# --- Import cascading utils ---
spec_cascade = importlib.util.spec_from_file_location("cascading_utils", "cascading_utils.py")
cascading_utils = importlib.util.module_from_spec(spec_cascade)
spec_cascade.loader.exec_module(cascading_utils)

# --- Streamlit page ---
st.set_page_config(page_title="Cascading Impact Simulation", layout="wide")
st.title("Cascading Impact Simulation (Live/Demo)")

# --- Sidebar: Data source ---
data_source = st.sidebar.radio('Data Source', ['API (Live)', 'Demo (Synthetic)'])
delay_minutes = st.sidebar.slider('Simulated delay (minutes)',
                                  min_value=5 if data_source=='API (Live)' else 10,
                                  max_value=120,
                                  value=30 if data_source=='API (Live)' else 70,
                                  step=5)

api_key = "e9304166bbmshe91c11acc615dfep11bc46jsn2f8ae8f1a57f"
api_host = "aerodatabox.p.rapidapi.com"
schedule_df = None
# --- Helper function ---
def validate_time_format(time_str):
    try:
        datetime.strptime(time_str, '%Y-%m-%dT%H:%M')
        return True
    except ValueError:
        return False

# --- Load schedule ---
if data_source == 'API (Live)':
    st.sidebar.header('AeroDataBox Live Arrivals/Departures')
    airport_iata = st.sidebar.text_input('Airport IATA code', value='FRA')
    start_time = st.sidebar.text_input('Start time (YYYY-MM-DDTHH:MM)', value=datetime.now(timezone.utc).strftime('%Y-%m-%dT00:00'))
    end_time = st.sidebar.text_input('End time (YYYY-MM-DDTHH:MM)', value=datetime.now(timezone.utc).strftime('%Y-%m-%dT23:59'))

    if airport_iata:
        if not validate_time_format(start_time) or not validate_time_format(end_time):
            st.error('Start and end time must be in format YYYY-MM-DDTHH:MM')
        else:
            st.write(f"Fetching arrivals and departures for {airport_iata}...")
            url = f"https://{api_host}/flights/airports/iata/{airport_iata}/{start_time}/{end_time}"
            headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": api_host}
            params = {
                "withLeg": "true", "direction": "Both", "withCancelled": "true",
                "withCodeshared": "true", "withCargo": "true", "withPrivate": "true",
                "withLocation": "false"
            }
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code == 200:
                    flights = resp.json()
                    all_flights = flights.get('arrivals', []) + flights.get('departures', [])
                    if all_flights:
                        schedule_df = pd.DataFrame(all_flights)

                        # --- Map fields ---
                        schedule_df['tail_number'] = schedule_df.get('aircraft', {}).apply(lambda x: x.get('reg', '') if isinstance(x, dict) else '') if 'aircraft' in schedule_df.columns else ''
                        def extract_time(row, keys=['scheduledTimeLocal','scheduledTime','actualTimeLocal','actualTime']):
                            if not isinstance(row, dict): return ''
                            for key in keys:
                                t = row.get(key, '')
                                if isinstance(t, dict):
                                    if t.get('local',''): return str(t['local'])
                                    if t.get('utc',''): return str(t['utc'])
                                elif t: return str(t)
                            return ''
                        schedule_df['STD'] = schedule_df['departure'].apply(extract_time) if 'departure' in schedule_df.columns else ''
                        schedule_df['STA'] = schedule_df['arrival'].apply(extract_time) if 'arrival' in schedule_df.columns else ''
                        schedule_df['From'] = schedule_df['departure'].apply(lambda x: x.get('airport', {}).get('icao','') if isinstance(x, dict) and 'airport' in x else '') if 'departure' in schedule_df.columns else ''
                        schedule_df['To'] = schedule_df['arrival'].apply(lambda x: x.get('airport', {}).get('icao','') if isinstance(x, dict) and 'airport' in x else '') if 'arrival' in schedule_df.columns else ''
                        schedule_df['callsign'] = schedule_df['number'] if 'number' in schedule_df.columns else ''

                        schedule_df['sched_dep'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STD'].astype(str), errors='coerce')
                        schedule_df['actual_arr'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STA'].astype(str), errors='coerce')

                        st.subheader('Arrivals/Departures (API)')
                        st.dataframe(schedule_df[['callsign','Date','From','To','tail_number','STD','STA']])
                    else:
                        st.warning("No flights returned from API.")
                else:
                    st.error(f"AeroDataBox API error: {resp.status_code}")
            except Exception as e:
                st.error(f"API fetch error: {e}")

elif data_source == 'Demo (Synthetic)':
    st.write('Using synthetic demo schedule.')
    demo_data = []
    base_hour = 8
    num_tails = 6
    flights_per_tail = 5
    for t in range(num_tails):
        tail = f'DEMO-TN-{t}'
        for f in range(flights_per_tail):
            dep_min = base_hour*60 + f*40 + t*3
            arr_min = dep_min - 60
            dep_hour, dep_minute = divmod(dep_min, 60)
            arr_hour, arr_minute = divmod(arr_min, 60)
            demo_data.append({
                'callsign': f'DEMO{t*flights_per_tail+f}',
                'Date': datetime.now(timezone.utc).date(),
                'From': f'AP{t*flights_per_tail+f}',
                'To': f'AP{t*flights_per_tail+f+1}',
                'tail_number': tail,
                'STD': f'{dep_hour:02d}:{dep_minute:02d}',
                'STA': f'{arr_hour:02d}:{arr_minute:02d}'
            })
    schedule_df = pd.DataFrame(demo_data)
    schedule_df['sched_dep'] = pd.to_datetime(schedule_df['Date'].astype(str)+' '+schedule_df['STD'].astype(str), errors='coerce')
    schedule_df['actual_arr'] = pd.to_datetime(schedule_df['Date'].astype(str)+' '+schedule_df['STA'].astype(str), errors='coerce')
    st.subheader('Arrivals/Departures (Demo)')
    st.dataframe(schedule_df[['callsign','Date','From','To','tail_number','STD','STA']])

# --- Common: Turnaround detection ---
turnaround_pairs = []
for tail in schedule_df['tail_number'].unique():
    flights_tail = schedule_df[schedule_df['tail_number']==tail].sort_values('sched_dep')
    for i in range(len(flights_tail)-1):
        arr_time = flights_tail.iloc[i]['actual_arr']
        next_dep_time = flights_tail.iloc[i+1]['sched_dep']
        if pd.notna(arr_time) and pd.notna(next_dep_time):
            turnaround = (next_dep_time - arr_time).total_seconds()/60
            if 0 < turnaround <= 120:
                turnaround_pairs.append((flights_tail.index[i], flights_tail.index[i+1], tail, turnaround))
if turnaround_pairs:
    st.success(f"Found {len(turnaround_pairs)} turnaround connections for cascading simulation.")
else:
    st.warning("No valid turnaround connections found.")

# --- Cascading simulation ---
scores = []
for idx in schedule_df.index:
    affected_flights, _, _, new_df = cascading_utils.simulate_cascading_delay(schedule_df, idx, delay_minutes)
    total_delay = 0
    for f in affected_flights:
        if f != idx:
            orig_arr = schedule_df.at[f, 'actual_arr']
            new_arr = new_df.at[f, 'actual_arr']
            if pd.notna(orig_arr) and pd.notna(new_arr):
                total_delay += max((new_arr - orig_arr).total_seconds()/60, 0)
    criticality_score = len(affected_flights) * total_delay
    scores.append((idx, len(affected_flights), total_delay, criticality_score))

top_scores = sorted(scores, key=lambda x: x[3], reverse=True)[:10]
st.subheader('Top 10 Cascade-Critical Flights')
if top_scores:
    top_df = schedule_df.loc[[i[0] for i in top_scores]].copy()
    top_df['affected_flights'] = [i[1] for i in top_scores]
    top_df['total_propagated_delay'] = [i[2] for i in top_scores]
    top_df['criticality_score'] = [i[3] for i in top_scores]
    st.dataframe(top_df[['callsign','Date','From','To','tail_number','STD','STA','affected_flights','total_propagated_delay','criticality_score']])
else:
    st.info('No flights found for cascade analysis.')

# --- Optimization (common for API and Demo) ---
st.subheader('Schedule Optimization')
if st.button('Optimize Schedule'):
    opt_result = schedule_optimizer.optimize_schedule(schedule_df, runway_capacity=10, shift_window=120, turnaround_min=30)
    if opt_result is not None:
        st.success('Optimization complete!')
        show_cols = ['callsign','tail_number','STD','optimized_sched_dep','delay_minutes']
        st.dataframe(opt_result[show_cols].rename(columns={'STD':'Original STD','optimized_sched_dep':'Optimized STD','delay_minutes':'Shift (min)'}))
        total_opt_delay = opt_result['delay_minutes'].clip(lower=0).sum()
        st.info(f"Total delay after optimization: {int(total_opt_delay)} minutes.")

        # --- Highlight large shifts ---
        large_shift_threshold = 60
        opt_result['shift_minutes'] = (opt_result['optimized_sched_dep'] - opt_result['sched_dep']).dt.total_seconds()/60
        large_shift_flights = opt_result[opt_result['shift_minutes'].abs() > large_shift_threshold]
        if not large_shift_flights.empty:
            st.subheader(f"Flights with shifts > {large_shift_threshold} minutes")
            st.dataframe(large_shift_flights[['callsign','tail_number','STD','optimized_sched_dep','shift_minutes']].rename(
                columns={'STD':'Original STD','optimized_sched_dep':'Optimized STD','shift_minutes':'Shift (min)'}
            ))
        else:
            st.info(f"No flights shifted more than {large_shift_threshold} minutes.")

        # --- Scatter plot ---
        df = opt_result.copy()
        plt.figure(figsize=(14,6))
        normal_flights = df[df['shift_minutes'].abs() <= 60]
        plt.scatter(normal_flights['sched_dep'], normal_flights['callsign'], color='red', label='Original Departure', s=50)
        plt.scatter(normal_flights['optimized_sched_dep'], normal_flights['callsign'], color='green', label='Optimized Departure', s=50)
        critical_flights = df[df['shift_minutes'].abs() > 60]
        plt.scatter(critical_flights['sched_dep'], critical_flights['callsign'], color='orange', label='Original Departure (Large Shift)', s=80, marker='X')
        plt.scatter(critical_flights['optimized_sched_dep'], critical_flights['callsign'], color='purple', label='Optimized Departure (Large Shift)', s=80, marker='X')
        for idx, row in df.iterrows():
            plt.plot([row['sched_dep'], row['optimized_sched_dep']], [row['callsign'], row['callsign']], color='blue', alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('Flights (Callsign)')
        plt.title('Original vs Optimized Flight Departures (Highlighted Large Shifts)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.error('No feasible optimization found.')
