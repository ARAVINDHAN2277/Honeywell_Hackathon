
import streamlit as st
import pandas as pd

import importlib.util
import requests
from datetime import datetime, timedelta, UTC
import matplotlib.pyplot as plt
import networkx as nx
# Import optimizer
import importlib.util as _importlib_util
spec_opt = _importlib_util.spec_from_file_location("optimize_schedule", "schedule_optimizer.py")
schedule_optimizer = _importlib_util.module_from_spec(spec_opt)
spec_opt.loader.exec_module(schedule_optimizer)


st.set_page_config(page_title="Cascading Impact Simulation", layout="wide")
st.title("Cascading Impact Simulation (Live from AeroDataBox)")

# Import cascading utils
spec_cascade = importlib.util.spec_from_file_location("cascading_utils", "cascading_utils.py")
cascading_utils = importlib.util.module_from_spec(spec_cascade)
spec_cascade.loader.exec_module(cascading_utils)


def visualize_cascade(schedule_df, turnaround_threshold, title='Cascading Flight Graph'):
	"""Build and render a professional cascading graph and a diagnostic table in Streamlit."""
	G, critical_flights = cascading_utils.build_cascading_graph(schedule_df, min_turn_required=turnaround_threshold)
	st.subheader(title)
	if G.number_of_nodes() == 0:
		st.info('No cascading connections to visualize with the current threshold.')
		return
	# layout and styling
	plt.style.use('seaborn-whitegrid')
	pos = nx.kamada_kawai_layout(G)
	out_degrees = dict(G.out_degree())
	max_out = max(out_degrees.values()) if out_degrees else 1
	node_colors = [out_degrees.get(n, 0) for n in G.nodes()]
	node_sizes = [320 + 1100 * (out_degrees.get(n, 0) / max_out) for n in G.nodes()]

	# edge widths scaled by tightness of turnaround
	edge_widths = []
	for u, v in G.edges():
		turn = G[u][v].get('turnaround', None)
		if turn is None:
			edge_widths.append(1.0)
		else:
			score = max(0.0, (turnaround_threshold - turn) / max(1.0, turnaround_threshold))
			edge_widths.append(0.8 + 6.0 * score)

	fig, ax = plt.subplots(figsize=(13, 8))
	nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.plasma, alpha=0.95, linewidths=0.9, edgecolors='k')
	edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=16, edge_color='gray', width=edge_widths, alpha=0.9)
	labels = {n: (schedule_df.at[n, 'callsign'] if 'callsign' in schedule_df.columns and pd.notna(schedule_df.at[n, 'callsign']) else schedule_df.at[n, 'tail_number']) for n in G.nodes()}
	nx.draw_networkx_labels(G, pos, labels, font_size=9)

	sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
	cbar.set_label('Out-degree (impact)')

	from matplotlib.lines import Line2D
	legend_elements = [Line2D([0], [0], color='gray', lw=1, label='Low cascade risk'), Line2D([0], [0], color='gray', lw=4, label='High cascade risk')]
	ax.legend(handles=legend_elements, loc='upper right')
	ax.set_title('Cascading Flight Graph (edges show tight turnarounds)')
	ax.axis('off')
	st.pyplot(fig)

	# Diagnostic table: show turnarounds that created edges
	edge_rows = []
	for u, v, data in G.edges(data=True):
		edge_rows.append({
			'from_idx': u,
			'to_idx': v,
			'from_callsign': schedule_df.at[u, 'callsign'] if 'callsign' in schedule_df.columns else '',
			'to_callsign': schedule_df.at[v, 'callsign'] if 'callsign' in schedule_df.columns else '',
			'tail_number': schedule_df.at[u, 'tail_number'],
			'turnaround_min': round(data.get('turnaround', 0), 1)
		})
	if edge_rows:
		st.markdown('**Turnarounds (minutes) that created edges:**')
		st.dataframe(pd.DataFrame(edge_rows))

	# Explain columns and logic briefly
	st.markdown('**Notes:** Turnaround = next scheduled departure - previous actual arrival (minutes).')
	st.markdown('Edges are created when Turnaround < Turnaround threshold. Node size/color = out-degree (how many later flights it can affect).')

	# Example: explain a sample row format
	st.markdown('**Sample row explanation:**')
	st.markdown('- Index: internal row index used in graph nodes')
	st.markdown("- callsign: flight identifier (e.g., 'DEMO0')")
	st.markdown('- Date: flight date')
	st.markdown('- From / To: origin and destination')
	st.markdown('- tail_number: aircraft registration or tail used to link flights')
	st.markdown('- STD / STA: scheduled departure/arrival local times (HH:MM)')
	st.markdown('- turnaround_min: computed for edges; small values (e.g., 0-30) are risky')
	st.markdown('For example, the row `0 DEMO0 2025-08-24 AP0 AP1 DEMO-TN-0 08:00 07:00 5 930 4650` can be read as: index=0, callsign=DEMO0, date=2025-08-24, origin AP0, dest AP1, tail DEMO-TN-0, scheduled dep 08:00, scheduled arr 07:00 — the trailing numbers are dataset-specific metrics (e.g., slot counts or encoded fields) and can be inspected in the dataframe shown above.')




# --- Data source selection ---

data_source = st.sidebar.radio('Data Source', ['API (Live)', 'Demo (Synthetic)'])
if data_source == 'Demo (Synthetic)':
	delay_minutes = st.sidebar.slider('Simulated delay (minutes)', min_value=10, max_value=120, value=70, step=5)
else:
	delay_minutes = st.sidebar.slider('Simulated delay (minutes)', min_value=5, max_value=120, value=30, step=5)

# Turnaround threshold (minutes) controls when an edge (tight turnaround) is created
turnaround_threshold = st.sidebar.slider('Turnaround threshold (minutes)', min_value=0, max_value=180, value=45, step=5)

if data_source == 'API (Live)':
	st.sidebar.header('AeroDataBox Live Arrivals/Departures')
	st.sidebar.markdown('**Use IATA code (e.g., FRA for Frankfurt, LHR for London Heathrow, JFK for New York JFK).**')
	airport_iata = st.sidebar.text_input('Airport IATA code', value='FRA')
	start_time = st.sidebar.text_input('Start time (YYYY-MM-DDTHH:MM)', value=datetime.now(UTC).strftime('%Y-%m-%dT00:00'))
	end_time = st.sidebar.text_input('End time (YYYY-MM-DDTHH:MM)', value=datetime.now(UTC).strftime('%Y-%m-%dT23:59'))
else:
	airport_iata = 'DEMO'
	start_time = None
	end_time = None

api_key = "e9304166bbmshe91c11acc615dfep11bc46jsn2f8ae8f1a57f"
api_host = "aerodatabox.p.rapidapi.com"


def validate_time_format(time_str):
	try:
		datetime.strptime(time_str, '%Y-%m-%dT%H:%M')
		return True
	except ValueError:
		return False


# --- Main logic for each data source ---
if data_source == 'API (Live)':
	if airport_iata:
		if not validate_time_format(start_time) or not validate_time_format(end_time):
			st.error('Start and end time must be in format YYYY-MM-DDTHH:MM (e.g., 2025-08-23T00:00)')
		else:
			st.write(f"Fetching arrivals and departures for {airport_iata} from AeroDataBox...")
			url = f"https://{api_host}/flights/airports/iata/{airport_iata}/{start_time}/{end_time}"
			params = {
				"withLeg": "true",
				"direction": "Both",
				"withCancelled": "true",
				"withCodeshared": "true",
				"withCargo": "true",
				"withPrivate": "true",
				"withLocation": "false"
			}
			headers = {
				"x-rapidapi-key": api_key,
				"x-rapidapi-host": api_host
			}
			try:
				resp = requests.get(url, headers=headers, params=params, timeout=30)
				if resp.status_code == 200:
					flights = resp.json()
					arrivals = flights.get('arrivals', [])
					departures = flights.get('departures', [])
					all_flights = arrivals + departures
					if all_flights:
						schedule_df = pd.DataFrame(all_flights)
						# ...existing mapping and simulation code...
						# (PASTE the mapping and simulation code block here)
						# --- Map AeroDataBox fields to expected columns ---
						schedule_df['tail_number'] = schedule_df.get('aircraft', {}).apply(lambda x: x.get('reg', '') if isinstance(x, dict) else '') if 'aircraft' in schedule_df.columns else ''
						schedule_df['Date'] = pd.to_datetime(schedule_df['departure'].apply(lambda x: x.get('scheduledTimeLocal', '') if isinstance(x, dict) else ''), errors='coerce').dt.date if 'departure' in schedule_df.columns else ''
						def extract_best_time(row, keys=['scheduledTimeLocal','scheduledTime','actualTimeLocal','actualTime']):
							# Try all keys, prefer 'local' if dict, always return string
							if not isinstance(row, dict):
								return ''
							for key in keys:
								t = row.get(key, '')
								if isinstance(t, dict):
									if t.get('local', ''):
										return str(t['local'])
									if t.get('utc', ''):
										return str(t['utc'])
								elif t:
									return str(t)
							return ''
						schedule_df['STD'] = schedule_df['departure'].apply(lambda x: extract_best_time(x)) if 'departure' in schedule_df.columns else ''
						schedule_df['STA'] = schedule_df['arrival'].apply(lambda x: extract_best_time(x)) if 'arrival' in schedule_df.columns else ''
						# Ensure all values are strings (never dicts or None)
						# Final cleanup: ensure all STD/STA values are plain strings (never dict, list, None, etc.)
						def force_string(val):
							if isinstance(val, (dict, list, tuple)) or val is None:
								return ''
							return str(val)
						schedule_df['STD'] = schedule_df['STD'].apply(force_string).astype(str)
						schedule_df['STA'] = schedule_df['STA'].apply(force_string).astype(str)
						schedule_df['From'] = schedule_df['departure'].apply(lambda x: x.get('airport', {}).get('icao', '') if isinstance(x, dict) and 'airport' in x else '') if 'departure' in schedule_df.columns else ''
						schedule_df['To'] = schedule_df['arrival'].apply(lambda x: x.get('airport', {}).get('icao', '') if isinstance(x, dict) and 'airport' in x else '') if 'arrival' in schedule_df.columns else ''
						schedule_df['callsign'] = schedule_df['number'] if 'number' in schedule_df.columns else ''
						schedule_df['sched_dep'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STD'].astype(str), errors='coerce')
						schedule_df['actual_arr'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STA'].astype(str), errors='coerce')
						st.subheader(f'Arrivals/Departures at {airport_iata}')
						st.dataframe(schedule_df[['callsign','Date','From','To','tail_number','STD','STA']])
						missing_std = schedule_df['STD'].isna().sum()
						missing_sta = schedule_df['STA'].isna().sum()
						if missing_std > 0 or missing_sta > 0:
							st.warning(f"{missing_std} flights missing STD and {missing_sta} flights missing STA. Some flights may not have scheduled time info in the source data.")
						# --- Turnaround connection detection ---
						turnaround_pairs = []
						for tail in schedule_df['tail_number'].unique():
							flights_tail = schedule_df[schedule_df['tail_number'] == tail].sort_values('sched_dep')
							for i in range(len(flights_tail) - 1):
								arr_time = flights_tail.iloc[i]['actual_arr']
								next_dep_time = flights_tail.iloc[i+1]['sched_dep']
								if pd.notna(arr_time) and pd.notna(next_dep_time):
									turnaround = (next_dep_time - arr_time).total_seconds() / 60
									if 0 < turnaround <= 120:
										turnaround_pairs.append((flights_tail.index[i], flights_tail.index[i+1], tail, turnaround))
						if turnaround_pairs:
							st.success(f"Found {len(turnaround_pairs)} turnaround connections (arrival followed by departure within 2 hours for same tail number). Cascading impact can be simulated.")
						else:
							st.warning("No valid turnaround connections found. Cascading impact will not propagate.")
						# --- Cascading simulation ---
						scores = []
						for idx in schedule_df.index:
							affected_flights, _, _, new_df = cascading_utils.simulate_cascading_delay(schedule_df, idx, delay_minutes, min_turn_required=turnaround_threshold)
							total_delay = 0
							for f in affected_flights:
								if f != idx:
									orig_arr = schedule_df.at[f, 'actual_arr']
									new_arr = new_df.at[f, 'actual_arr']
									if pd.notna(orig_arr) and pd.notna(new_arr):
										delay = (new_arr - orig_arr).total_seconds() / 60
										total_delay += max(delay, 0)
							criticality_score = len(affected_flights) * total_delay
							scores.append((idx, len(affected_flights), total_delay, criticality_score))
						top_scores = sorted(scores, key=lambda x: x[3], reverse=True)[:10]
						st.subheader('Top 10 Cascade-Critical Flights (Live)')
						if top_scores:
							top_df = schedule_df.loc[[i[0] for i in top_scores]].copy()
							top_df['affected_flights'] = [i[1] for i in top_scores]
							top_df['total_propagated_delay'] = [i[2] for i in top_scores]
							top_df['criticality_score'] = [i[3] for i in top_scores]
							st.dataframe(top_df[['callsign','Date','From','To','tail_number','STD','STA','affected_flights','total_propagated_delay','criticality_score']])
							# provide download for top cascade table
							csv_bytes = top_df.to_csv(index=False).encode('utf-8')
							st.download_button('Download Top-10 cascade table (CSV)', data=csv_bytes, file_name='top10_cascade_live.csv', mime='text/csv')
							# save to outputs folder
							import os
							os.makedirs('outputs', exist_ok=True)
							with open(os.path.join('outputs','top10_cascade_live.csv'),'wb') as f:
								f.write(csv_bytes)
						else:
							st.info('No flights found for selected airport and date/time.')
							# --- Enhanced Cascading Graph Visualization ---
							import matplotlib.pyplot as plt
							import networkx as nx
							st.subheader('Cascading Impact Graph')
							G, critical_flights = cascading_utils.build_cascading_graph(schedule_df, min_turn_required=turnaround_threshold)
							if G.number_of_nodes() > 0:
								pos = nx.kamada_kawai_layout(G)
								out_degrees = dict(G.out_degree())
								node_sizes = [300 + 1000 * (out_degrees.get(n, 0)/max(1, max(out_degrees.values()))) for n in G.nodes()]
								node_colors = [out_degrees.get(n, 0) for n in G.nodes()]
								labels = {n: schedule_df.at[n, 'callsign'] if 'callsign' in schedule_df.columns and pd.notna(schedule_df.at[n, 'callsign']) else schedule_df.at[n, 'tail_number'] for n in G.nodes()}
								plt.figure(figsize=(12, 7))
								nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.85)
								edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', width=2)
								nx.draw_networkx_labels(G, pos, labels, font_size=10)
								plt.colorbar(nodes, label='Out-degree (Impact)')
								plt.axis('off')
								st.pyplot(plt)
							else:
								st.info('No cascading connections to visualize.')
						# --- Optimization Feature (API) ---
						st.subheader('Schedule Optimization (Live API)')
						if st.button('Optimize Schedule (API)'):
							opt_result = schedule_optimizer.optimize_schedule(
								schedule_df,
								runway_capacity=10,shift_window=120,turnaround_min=30 )
							if opt_result is not None:
								st.success('Optimization complete!')
								show_cols = ['callsign', 'tail_number', 'STD', 'optimized_sched_dep', 'delay_minutes']
								st.dataframe(opt_result[show_cols].rename(columns={'STD': 'Original STD','optimized_sched_dep': 'Optimized STD','delay_minutes': 'Shift (min)'}))
								total_opt_delay = opt_result['delay_minutes'].clip(lower=0).sum()
								st.info(f"Total delay after optimization: {int(total_opt_delay)} minutes.")
								 # --- Highlight flights with large shifts ---
								large_shift_threshold = 60  # minute
								opt_result['shift_minutes'] = (
            opt_result['optimized_sched_dep'] - opt_result['sched_dep']
        ).dt.total_seconds() / 60
								large_shift_flights = opt_result[opt_result['shift_minutes'].abs() > large_shift_threshold]
								if not large_shift_flights.empty:
									st.subheader(f"Flights with shifts > {large_shift_threshold} minutes")
									display_cols = ['callsign', 'tail_number', 'STD', 'optimized_sched_dep', 'shift_minutes']
									display_df = large_shift_flights[display_cols].copy()
									display_df = display_df.rename(columns={
                'STD': 'Original STD',
                'optimized_sched_dep': 'Optimized STD',
                'shift_minutes': 'Shift (min)'
            })
									st.dataframe(display_df)
									# download optimized schedule
									opt_csv = opt_result.to_csv(index=False).encode('utf-8')
									st.download_button('Download optimized schedule (CSV)', data=opt_csv, file_name='optimized_schedule_live.csv', mime='text/csv')
									with open(os.path.join('outputs','optimized_schedule_live.csv'),'wb') as f:
										f.write(opt_csv)
								else:
									st.info(f"No flights shifted more than {large_shift_threshold} minutes.")
							else:
								st.error('No feasible optimization found. Try increasing the shift window or runway capacity.')

					else:
						st.info('No arrivals or departures found for this airport and time range.')
				else:
					st.error(f'AeroDataBox API error: {resp.status_code} - {resp.text}')
			except Exception as e:
				st.error(f'Error fetching data from AeroDataBox: {e}')
	else:
		st.info('Enter an airport IATA code to begin analysis.')

elif data_source == 'Demo (Synthetic)':
	st.write('Using synthetic demo schedule for cascading simulation.')
	# Generate spaced synthetic schedule for guaranteed optimization
	demo_data = []
	base_hour = 8
	num_tails = 6
	flights_per_tail = 5  # more flights for denser schedule
	for t in range(num_tails):
		tail = f'DEMO-TN-{t}'
		for f in range(flights_per_tail):
			# Flights depart every 40 min, so turnarounds are tighter
			dep_min = base_hour * 60 + f * 40 + t * 3  # denser, slight stagger
			arr_min = dep_min - 60  # 1 hour before
			dep_hour, dep_minute = divmod(dep_min, 60)
			arr_hour, arr_minute = divmod(arr_min, 60)
			demo_data.append({
				'callsign': f'DEMO{t*flights_per_tail+f}',
				'Date': datetime.now(UTC).date(),
				'From': f'AP{t*flights_per_tail+f}',
				'To': f'AP{t*flights_per_tail+f+1}',
				'tail_number': tail,
				'STD': f'{dep_hour:02d}:{dep_minute:02d}',
				'STA': f'{arr_hour:02d}:{arr_minute:02d}'
			})
	schedule_df = pd.DataFrame(demo_data)
	schedule_df['sched_dep'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STD'].astype(str), errors='coerce')
	schedule_df['actual_arr'] = pd.to_datetime(schedule_df['Date'].astype(str) + ' ' + schedule_df['STA'].astype(str), errors='coerce')
	st.subheader('Arrivals/Departures (Demo)')
	st.dataframe(schedule_df[['callsign','Date','From','To','tail_number','STD','STA']])
	# --- Turnaround connection detection ---
	turnaround_pairs = []
	for tail in schedule_df['tail_number'].unique():
		flights_tail = schedule_df[schedule_df['tail_number'] == tail].sort_values('sched_dep')
		for i in range(len(flights_tail) - 1):
			arr_time = flights_tail.iloc[i]['actual_arr']
			next_dep_time = flights_tail.iloc[i+1]['sched_dep']
			if pd.notna(arr_time) and pd.notna(next_dep_time):
				turnaround = (next_dep_time - arr_time).total_seconds() / 60
				if 0 < turnaround <= 120:
					turnaround_pairs.append((flights_tail.index[i], flights_tail.index[i+1], tail, turnaround))
	if turnaround_pairs:
		st.success(f"Found {len(turnaround_pairs)} turnaround connections (arrival followed by departure within 2 hours for same tail number). Cascading impact can be simulated.")
	else:
		st.warning("No valid turnaround connections found. Cascading impact will not propagate.")
	# --- Cascading simulation ---
	scores = []
	for idx in schedule_df.index:
		affected_flights, _, _, new_df = cascading_utils.simulate_cascading_delay(schedule_df, idx, delay_minutes, min_turn_required=turnaround_threshold)
		total_delay = 0
		for f in affected_flights:
			if f != idx:
				orig_arr = schedule_df.at[f, 'actual_arr']
				new_arr = new_df.at[f, 'actual_arr']
				if pd.notna(orig_arr) and pd.notna(new_arr):
					delay = (new_arr - orig_arr).total_seconds() / 60
					total_delay += max(delay, 0)
		criticality_score = len(affected_flights) * total_delay
		scores.append((idx, len(affected_flights), total_delay, criticality_score))
	top_scores = sorted(scores, key=lambda x: x[3], reverse=True)[:10]
	st.subheader('Top 10 Cascade-Critical Flights (Demo)')
	if top_scores:
		top_df = schedule_df.loc[[i[0] for i in top_scores]].copy()
		top_df['affected_flights'] = [i[1] for i in top_scores]
		top_df['total_propagated_delay'] = [i[2] for i in top_scores]
		top_df['criticality_score'] = [i[3] for i in top_scores]
		st.dataframe(top_df[['callsign','Date','From','To','tail_number','STD','STA','affected_flights','total_propagated_delay','criticality_score']])
		csv_bytes = top_df.to_csv(index=False).encode('utf-8')
		st.download_button('Download Top-10 cascade table (CSV)', data=csv_bytes, file_name='top10_cascade_demo.csv', mime='text/csv')
		import os
		os.makedirs('outputs', exist_ok=True)
		with open(os.path.join('outputs','top10_cascade_demo.csv'),'wb') as f:
			f.write(csv_bytes)
		if all(i[1] <= 1 or i[2] == 0 or i[3] == 0 for i in top_scores):
			st.info('Cascade impact is minimal (no multi-flight propagation or delay). Try increasing the simulated delay for more effect.')
	else:
		st.warning('No flights found for cascade analysis.')

	# --- Enhanced Cascading Graph Visualization ---
	st.subheader('Cascading Impact Graph')
	G, critical_flights = cascading_utils.build_cascading_graph(schedule_df, min_turn_required=turnaround_threshold)
	if G.number_of_nodes() > 0:
		pos = nx.kamada_kawai_layout(G)
		out_degrees = dict(G.out_degree())
		max_out = max(out_degrees.values()) if out_degrees else 1
		node_sizes = [300 + 1000 * (out_degrees.get(n, 0)/max(1, max_out)) for n in G.nodes()]
		node_colors = [out_degrees.get(n, 0) for n in G.nodes()]
		labels = {n: schedule_df.at[n, 'callsign'] if 'callsign' in schedule_df.columns and pd.notna(schedule_df.at[n, 'callsign']) else schedule_df.at[n, 'tail_number'] for n in G.nodes()}
		plt.figure(figsize=(12, 7))
		nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.85)
		edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', width=2)
		nx.draw_networkx_labels(G, pos, labels, font_size=10)
		plt.colorbar(nodes, label='Out-degree (Impact)')
		plt.axis('off')
		st.pyplot(plt)
	else:
		st.info('No cascading connections to visualize.')
	
	# --- Optimization Feature ---
	st.subheader('Schedule Optimization (Demo)')
	if st.button('Optimize Schedule'):
		opt_result = schedule_optimizer.optimize_schedule(schedule_df, runway_capacity=10, shift_window=120, turnaround_min=30)
		if opt_result is not None:
			st.success('Optimization complete!')
			show_cols = ['callsign', 'tail_number', 'STD', 'optimized_sched_dep', 'delay_minutes']
			st.dataframe(opt_result[show_cols].rename(columns={
				'STD': 'Original STD',
				'optimized_sched_dep': 'Optimized STD',
				'delay_minutes': 'Shift (min)'
			}))
			total_opt_delay = opt_result['delay_minutes'].clip(lower=0).sum()
			st.info(f"Total delay after optimization: {int(total_opt_delay)} minutes.")

			# --- Highlight flights with large shifts ---
			large_shift_threshold = 60  # minutes
			opt_result['shift_minutes'] = (opt_result['optimized_sched_dep'] - opt_result['sched_dep']).dt.total_seconds() / 60
			large_shift_flights = opt_result[opt_result['shift_minutes'].abs() > large_shift_threshold]
			if not large_shift_flights.empty:
				st.subheader(f"Flights with shifts > {large_shift_threshold} minutes")
				display_cols = ['callsign', 'tail_number', 'STD', 'optimized_sched_dep', 'shift_minutes']
				display_df = large_shift_flights[display_cols].copy()
				display_df = display_df.rename(columns={
					'STD': 'Original STD',
					'optimized_sched_dep': 'Optimized STD',
					'shift_minutes': 'Shift (min)'
				})
				st.dataframe(display_df)
			else:
				st.info(f"No flights shifted more than {large_shift_threshold} minutes.")
		else:
			st.error('No feasible optimization found. Try increasing the shift window or runway capacity.')
