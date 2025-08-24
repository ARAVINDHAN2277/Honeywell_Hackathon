import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import requests
import os
from datetime import datetime
import networkx as nx
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        # configuration may fail in some environments; API calls will surface errors
        pass


def ask_gemini(query):
    if not GEMINI_API_KEY:
        return 'Gemini API key not configured. Place GEMINI_API_KEY in a .env file or environment.'
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(query)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Gemini API error: {e}"

# Import utilities
import importlib.util
spec = importlib.util.spec_from_file_location("cascading_utils", "cascading_utils.py")
cascading_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cascading_utils)

spec2 = importlib.util.spec_from_file_location("schedule_optimizer", "schedule_optimizer.py")
schedule_optimizer = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(schedule_optimizer)

st.set_page_config(page_title="Flight Ops Toolkit", layout="wide")
st.title("Flight Schedule Optimization")

# Helper: load schedule from upload or default
def load_schedule(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Cannot read uploaded file: {e}")
                return None
    else:
        fp = 'flight.xlsx'
        if os.path.exists(fp):
            try:
                # try reading first sheet or whole excel
                df = pd.read_excel(fp)
            except Exception as e:
                st.error(f"Failed to read {fp}: {e}")
                return None
        else:
            st.info('No default flight.xlsx found. Upload a schedule (Excel/CSV).')
            return None

    # Basic parsing: try to create sched_dep, sched_arr, actual_dep, actual_arr
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Extract tail registration into a canonical 'tail_number' column when possible.
    # Handles values like: "A20N (VT-EXU)" -> tail_number = "VT-EXU"
    try:
        tail_source = None
        for c in df.columns:
            cl = c.lower()
            if 'tail' in cl or 'registration' in cl or 'aircraft' in cl or 'reg' in cl:
                tail_source = c
                break
        if tail_source:
            def extract_tail(val):
                try:
                    if pd.isna(val):
                        return ''
                    s = str(val).strip()
                    m = re.search(r"\(([^)]+)\)", s)
                    if m:
                        return m.group(1).strip()
                    # if format like "A20N - VT-EXU" or "A20N VT-EXU"
                    parts = re.split(r"[-\s]+", s)
                    # look for part with dash like VT-EXU or starting with VT
                    for p in parts[::-1]:
                        if re.match(r"^[A-Z]{2}-?[A-Z0-9]+$", p.upper()):
                            return p.strip()
                    return s
                except Exception:
                    return ''
            df['tail_number'] = df[tail_source].apply(extract_tail)
        else:
            # ensure column exists so downstream code can rely on it
            if 'tail_number' not in df.columns:
                df['tail_number'] = ''
    except Exception:
        if 'tail_number' not in df.columns:
            df['tail_number'] = ''

    # Ensure there is a 'callsign' column. Try common column names, else build a fallback.
    try:
        callsign_source = None
        for c in df.columns:
            cl = c.lower()
            if cl in ('callsign','flight','flight_no','flightno','flight_number','number','flightnum','flt') or 'callsign' in cl or 'flight' in cl:
                callsign_source = c
                break
        if callsign_source:
            df['callsign'] = df[callsign_source].astype(str)
        else:
            # fallback: use tail_number + index or index alone
            if 'tail_number' in df.columns and df['tail_number'].notna().any():
                df['callsign'] = df['tail_number'].astype(str) + '-' + df.index.astype(str)
            else:
                df['callsign'] = df.index.astype(str)
    except Exception:
        if 'callsign' not in df.columns:
            df['callsign'] = df.index.astype(str)

    # If Date and STD/STA columns exist
    def combine_date_time(row, date_col, time_col):
        date_val = row.get(date_col, None)
        time_val = row.get(time_col, None)
        if pd.isna(date_val) or pd.isna(time_val):
            return pd.NaT
        try:
            t = pd.to_datetime(str(time_val)).time()
            dt = pd.to_datetime(date_val).replace(hour=t.hour, minute=t.minute, second=t.second)
            return dt
        except Exception:
            try:
                return pd.to_datetime(str(date_val) + ' ' + str(time_val), errors='coerce')
            except Exception:
                return pd.NaT

    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        except Exception:
            pass
    # possible name variants
    if 'STD' in df.columns and 'Date' in df.columns:
        df['sched_dep'] = df.apply(lambda r: combine_date_time(r, 'Date', 'STD'), axis=1)
    if 'STA' in df.columns and 'Date' in df.columns:
        df['sched_arr'] = df.apply(lambda r: combine_date_time(r, 'Date', 'STA'), axis=1)
    if 'ATD' in df.columns and 'Date' in df.columns:
        df['actual_dep'] = df.apply(lambda r: combine_date_time(r, 'Date', 'ATD'), axis=1)
    if 'ATA' in df.columns and 'Date' in df.columns:
        df['actual_arr'] = df.apply(lambda r: combine_date_time(r, 'Date', 'ATA'), axis=1)

    # Fallback: try parsing sched_dep if present
    for col in ['sched_dep', 'sched_arr', 'actual_dep', 'actual_arr']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

# Sidebar: upload and parameters
st.sidebar.header('Input')
uploaded = st.sidebar.file_uploader('Upload schedule (Excel/CSV)', type=['xlsx','csv'])
use_demo = st.sidebar.checkbox('Use demo synthetic schedule', value=False)

st.sidebar.header('Cascade settings')
turnaround_threshold = st.sidebar.slider('Turnaround threshold (minutes)', min_value=0, max_value=180, value=45, step=5)
simulated_delay = st.sidebar.slider('Simulated delay (minutes)', min_value=0, max_value=240, value=60, step=5)

# Load data
if use_demo:
    # build same demo as cascading_simulation
    demo_data = []
    base_hour = 8
    num_tails = 6
    flights_per_tail = 5
    for t in range(num_tails):
        tail = f'DEMO-TN-{t}'
        for f in range(flights_per_tail):
            dep_min = base_hour * 60 + f * 40 + t * 3
            arr_min = dep_min - 60
            dep_hour, dep_minute = divmod(dep_min, 60)
            arr_hour, arr_minute = divmod(arr_min, 60)
            demo_data.append({
                'callsign': f'DEMO{t*flights_per_tail+f}',
                'Date': datetime.now().date(),
                'From': f'AP{t*flights_per_tail+f}',
                'To': f'AP{t*flights_per_tail+f+1}',
                'tail_number': tail,
                'STD': f'{dep_hour:02d}:{dep_minute:02d}',
                'STA': f'{arr_hour:02d}:{arr_minute:02d}'
            })
    schedule_df = pd.DataFrame(demo_data)
    schedule_df['sched_dep'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STD'].astype(str), errors='coerce')
    schedule_df['actual_arr'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STA'].astype(str), errors='coerce')
else:
    schedule_df = load_schedule(uploaded)

if schedule_df is None:
    st.stop()

# --- NLP Query Interface (top of dashboard) ---
st.markdown('## Ask a question')

query = st.text_input('Chat: ask anything about flight operations, schedule, delays, or analytics:')
if query:
    q = query.lower()
    rule_based_answer = None
    # Rule-based: best time to takeoff/depart
    if ('best' in q and ('depart' in q or 'takeoff' in q)) or ('best time' in q and ('depart' in q or 'takeoff' in q)):
        if 'sched_dep' in schedule_df.columns and 'actual_dep' in schedule_df.columns:
            schedule_df['dep_delay'] = (schedule_df['actual_dep'] - schedule_df['sched_dep']).dt.total_seconds() / 60
            if schedule_df['dep_delay'].notna().any():
                schedule_df['dep_hour'] = schedule_df['sched_dep'].dt.hour
                avg_dep = schedule_df.groupby('dep_hour')['dep_delay'].mean().reset_index()
                best_row = avg_dep.loc[avg_dep['dep_delay'].idxmin()]
                fig, ax = plt.subplots(figsize=(8,3))
                sns.lineplot(x='dep_hour', y='dep_delay', data=avg_dep, marker='o', ax=ax)
                ax.set_xlabel('Scheduled departure hour')
                ax.set_ylabel('Average departure delay (min)')
                st.pyplot(fig)
                rule_based_answer = f"Best time to depart (lowest avg delay): {int(best_row['dep_hour']):02d}:00 with average delay {best_row['dep_delay']:.1f} min."
            else:
                rule_based_answer = 'Not enough actual departure times to compute delays.'
        else:
            rule_based_answer = 'Not enough actual/scheduled departure times to compute best time.'
    # Add more rule-based answers here as needed

    if rule_based_answer:
        st.markdown('Answer:')
        st.write(rule_based_answer)
    else:
        st.markdown(' Answer:')
        answer = ask_gemini(query)
        st.write(answer)

# Tabs
tabs = st.tabs(['Overview','Busiest slots','Best time','Delay predictor','Cascade','Optimizer'])

# Overview
with tabs[0]:
    st.header('Overview')
    st.write('Schedule preview:')
    # convert to strings for display to avoid pyarrow type conversion errors for mixed-type columns
    try:
        st.dataframe(schedule_df.head(200).astype(str))
    except Exception:
        st.dataframe(schedule_df.head(200))

# Busiest slots
with tabs[1]:
    st.header('Busiest Slots')
    # require sched_dep
    if 'sched_dep' not in schedule_df.columns or schedule_df['sched_dep'].isna().all():
        st.warning('No scheduled departure times available.')
    else:
        schedule_df['slot_15'] = schedule_df['sched_dep'].dt.floor('15min')
        schedule_df['slot_60'] = schedule_df['sched_dep'].dt.floor('60min')
        slot_counts = schedule_df.groupby('slot_15').size().reset_index(name='movements').sort_values('movements', ascending=False)
        st.subheader('Top 10 busiest 15-min slots')
        st.dataframe(slot_counts.head(10))
        # heatmap by hour
        schedule_df['hour'] = schedule_df['sched_dep'].dt.hour
        heat = schedule_df.groupby(['hour']).size()
        fig, ax = plt.subplots(figsize=(8,3))
    # Fix seaborn FutureWarning: assign hue and legend
    sns.barplot(x=heat.index, y=heat.values, ax=ax, palette='rocket', hue=heat.index, legend=False)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Movements')
    st.pyplot(fig)

# Best time
with tabs[2]:
    st.header('Best Time to Depart / Arrive')
    # need actual and sched
    if 'actual_dep' in schedule_df.columns and 'sched_dep' in schedule_df.columns:
        schedule_df['dep_delay'] = (schedule_df['actual_dep'] - schedule_df['sched_dep']).dt.total_seconds() / 60
    else:
        schedule_df['dep_delay'] = np.nan
    if 'actual_arr' in schedule_df.columns and 'sched_arr' in schedule_df.columns:
        schedule_df['arr_delay'] = (schedule_df['actual_arr'] - schedule_df['sched_arr']).dt.total_seconds() / 60
    else:
        schedule_df['arr_delay'] = np.nan
    if schedule_df['dep_delay'].notna().any():
        schedule_df['dep_hour'] = schedule_df['sched_dep'].dt.hour
        avg_dep = schedule_df.groupby('dep_hour')['dep_delay'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8,3))
        sns.lineplot(x='dep_hour', y='dep_delay', data=avg_dep, marker='o', ax=ax)
        ax.set_xlabel('Scheduled departure hour')
        ax.set_ylabel('Average departure delay (min)')
        st.pyplot(fig)
    else:
        st.info('Not enough actual departure times to compute delays.')

# Delay predictor (enhanced from predict_ui.py)
with tabs[3]:
    st.header('Delay Predictor')
    # dynamic import of slot traffic util
    try:
        spec_slot = importlib.util.spec_from_file_location('slot_traffic_utils', 'slot_traffic_utils.py')
        slot_traffic_utils = importlib.util.module_from_spec(spec_slot)
        spec_slot.loader.exec_module(slot_traffic_utils)
    except Exception:
        slot_traffic_utils = None

    # Cascading detection (optional)
    if schedule_df is not None and st.button('Detect Cascading Impact'):
        try:
            G, critical_flights = cascading_utils.build_cascading_graph(schedule_df)
            st.subheader('Top 10 Critical Flights (by disruption risk)')
            top_df = schedule_df.loc[[f[0] for f in critical_flights]]
            st.dataframe(top_df)
            st.write('Graph summary:')
            st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            st.write('Each edge represents a tight turnaround that may propagate delay.')
        except Exception as e:
            st.error(f"Error in cascading impact detection: {e}")

    # Input fields for a single flight
    date = st.date_input('Flight Date')
    std = st.text_input('Scheduled Departure Time (e.g., 06:00 or 8:14 AM)')
    atd = st.text_input('Actual Departure Time (e.g., 06:10 or 8:20 AM)')
    sta = st.text_input('Scheduled Arrival Time (e.g., 08:00 or 10:00 AM)')
    from_airport = st.text_input('From (e.g., BOM)')
    to_airport = st.text_input('To (e.g., DEL)')
    aircraft = st.text_input('Aircraft Type (e.g., A320)')

    # helper: combine date and time
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

    # weather features helper (reads weather_data.json if present)
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
        # feature engineering
        sched_dep = combine_date_time(date, std)
        actual_dep = combine_date_time(date, atd)
        sched_arr = combine_date_time(date, sta)

        dep_delay = (actual_dep - sched_dep).total_seconds() / 60 if pd.notna(actual_dep) and pd.notna(sched_dep) else 0
        hour = sched_dep.hour if pd.notna(sched_dep) else np.nan
        dayofweek = sched_dep.dayofweek if pd.notna(sched_dep) else np.nan
        route = f"{from_airport}-{to_airport}"
        aircraft_type = aircraft
        dep_slot_15min = sched_dep.floor('15min') if pd.notna(sched_dep) else pd.NaT

        # slot traffic
        congestion_details = ""
        origin_congestion = None
        dest_congestion = None
        if schedule_df is not None and pd.notna(sched_dep) and slot_traffic_utils is not None:
            try:
                slot_traffic = slot_traffic_utils.calculate_slot_traffic(schedule_df, from_airport, sched_dep)
                congestion_details += f"Origin slot traffic (from schedule): {slot_traffic} flights in this 15-min window.\n"
                origin_congestion = slot_traffic
            except Exception:
                congestion_details += "Origin slot traffic: unknown\n"
                origin_congestion = 1
        else:
            congestion_details += "Origin slot traffic: 1 (default/no schedule)\n"
            origin_congestion = 1

        # destination congestion best-effort
        if pd.notna(sched_arr):
            dest_congestion = 1
            congestion_details += "Destination slot traffic: best-effort default\n"

        with st.container():
            st.info(congestion_details)
            if origin_congestion is not None and origin_congestion > 10:
                st.warning('Warning: Origin runway/slot congestion is high!')
            if dest_congestion is not None and dest_congestion > 10:
                st.warning('Warning: Destination runway/slot congestion is high!')

        temperature_2m, windspeed_10m, precipitation, cloudcover = get_weather_features(sched_dep)

        # build input df
        input_dict = {
            'hour': [hour],
            'dayofweek': [dayofweek],
            'route': [route],
            'aircraft_type': [aircraft_type],
            'slot_traffic': [origin_congestion],
            'temperature_2m': [temperature_2m],
            'windspeed_10m': [windspeed_10m],
            'precipitation': [precipitation],
            'cloudcover': [cloudcover],
            'dep_delay': [dep_delay],
        }
        input_df = pd.DataFrame(input_dict)
        X_pred = pd.get_dummies(input_df)

        # load model and predict
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
            threshold = 0.2
            pred = int(prob > threshold)
            st.write(f"Predicted probability of delay: {prob:.2f}")
            st.write(f"Prediction (threshold {threshold}): {'Delayed' if pred else 'On Time'}")
        except Exception as e:
            st.info('Model or columns file not found. Place ensemble_model.pkl and ensemble_model_columns.pkl in repo or train a model.')
            st.write(str(e))

# Cascade
with tabs[4]:
    st.header('Cascading Impact')
    st.markdown('Use the turnaround threshold and simulated delay to explore cascades.')
    # call local visualize function using cascading_utils
    try:
        # build schedule for cascading_utils: ensure sched_dep and actual_arr exist
        if 'sched_dep' not in schedule_df.columns or 'actual_arr' not in schedule_df.columns:
            st.warning('sched_dep or actual_arr missing; cascading may not find edges. You can set demo mode or upload richer data.')
        # Build graph and visualize
        def local_visualize(schedule_df_local):
            G, critical = cascading_utils.build_cascading_graph(schedule_df_local, min_turn_required=turnaround_threshold)
            if G.number_of_nodes() == 0:
                st.info('No cascading connections found for current threshold.')
                return
            # draw similar to visualize_cascade
            pos = nx.kamada_kawai_layout(G)
            out_degrees = dict(G.out_degree())
            max_out = max(out_degrees.values()) if out_degrees else 0
            if max_out <= 0:
                max_out = 1
            node_colors = [out_degrees.get(n,0) for n in G.nodes()]
            node_sizes = [320 + 1100 * (out_degrees.get(n,0)/max_out) for n in G.nodes()]
            edge_widths = []
            for u,v in G.edges():
                turn = G[u][v].get('turnaround', None)
                if turn is None:
                    edge_widths.append(1.0)
                else:
                    # Avoid division by zero if threshold is zero
                    denom = max(1.0, turnaround_threshold)
                    score = max(0.0, (turnaround_threshold - turn) / denom)
                    edge_widths.append(0.8 + 6.0 * score)
            fig, ax = plt.subplots(figsize=(12,7))
            nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.plasma, alpha=0.95, linewidths=0.9, edgecolors='k')
            edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=14, edge_color='gray', width=edge_widths, alpha=0.9)
            labels = {n: (schedule_df_local.at[n,'callsign'] if 'callsign' in schedule_df_local.columns and pd.notna(schedule_df_local.at[n,'callsign']) else schedule_df_local.at[n,'tail_number']) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=9)
            st.pyplot(fig)
            # diagnostic
            edge_rows = []
            for u,v,data in G.edges(data=True):
                edge_rows.append({'from_idx':u,'to_idx':v,'from_callsign':schedule_df_local.at[u,'callsign'] if 'callsign' in schedule_df_local.columns else '', 'to_callsign':schedule_df_local.at[v,'callsign'] if 'callsign' in schedule_df_local.columns else '', 'tail_number':schedule_df_local.at[u,'tail_number'], 'turnaround_min':round(data.get('turnaround',0),1)})
            if edge_rows:
                st.dataframe(pd.DataFrame(edge_rows))
        # Compute Top-10 cascade-critical flights (same logic used in cascading_simulation.py)
        try:
            scores = []
            for idx in schedule_df.index:
                affected_flights, _, _, new_df = cascading_utils.simulate_cascading_delay(schedule_df, idx, simulated_delay, min_turn_required=turnaround_threshold)
                total_delay = 0
                for f in affected_flights:
                    if f != idx:
                        orig_arr = schedule_df.at[f, 'actual_arr'] if 'actual_arr' in schedule_df.columns else pd.NaT
                        new_arr = new_df.at[f, 'actual_arr'] if 'actual_arr' in new_df.columns else pd.NaT
                        if pd.notna(orig_arr) and pd.notna(new_arr):
                            delay = (new_arr - orig_arr).total_seconds() / 60
                            total_delay += max(delay, 0)
                criticality_score = len(affected_flights) * total_delay
                scores.append((idx, len(affected_flights), total_delay, criticality_score))
            top_scores = sorted(scores, key=lambda x: x[3], reverse=True)[:10]
            st.subheader('Top 10 Cascade-Critical Flights (tab)')
            if top_scores:
                top_df = schedule_df.loc[[i[0] for i in top_scores]].copy()
                top_df['affected_flights'] = [i[1] for i in top_scores]
                top_df['total_propagated_delay'] = [i[2] for i in top_scores]
                top_df['criticality_score'] = [i[3] for i in top_scores]
                display_cols = ['callsign','Date','From','To','tail_number','STD','STA','affected_flights','total_propagated_delay','criticality_score']
                st.dataframe(top_df[display_cols])
                csv_bytes = top_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Top-10 cascade table (CSV)', data=csv_bytes, file_name='top10_cascade_tab.csv', mime='text/csv')
                import os
                os.makedirs('outputs', exist_ok=True)
                with open(os.path.join('outputs','top10_cascade_tab.csv'),'wb') as f:
                    f.write(csv_bytes)
            else:
                st.info('No flights found for selected dataset or parameters.')
        except Exception as e:
            st.error(f'Failed to compute Top-10 cascade-critical flights: {e}')

        # Finally, show the visual graph
        local_visualize(schedule_df)
    except Exception as e:
        st.error(f'Error building cascade graph: {e}')

# Optimizer
with tabs[5]:
    st.header('Schedule Optimizer')
    st.markdown('Optimize departures subject to runway capacity and turnaround constraints.')
    run_opt = st.button('Run optimization')
    if run_opt:
        try:
            res = schedule_optimizer.optimize_schedule(schedule_df, runway_capacity=6, shift_window=60, turnaround_min=30)
            if res is None:
                st.error('No optimization result. Check inputs.')
            else:
                st.success('Optimization complete')
                st.dataframe(res.head(50))
        except Exception as e:
            st.error(f'Optimizer failed: {e}')
