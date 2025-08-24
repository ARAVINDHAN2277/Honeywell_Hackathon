import pandas as pd
import networkx as nx

def build_cascading_graph(schedule_df, min_turn_required=70):
    """
    Build a directed graph of flights showing cascading delay risk.
    Nodes: flights (by index or flight_id)
    Edges: from flight A to flight B if same aircraft and tight turnaround
    Returns: graph (networkx.DiGraph), critical_flights (list of top 10 by out-degree)
    """
    # Ensure required columns
    required_cols = ['tail_number', 'sched_dep', 'actual_arr']
    for col in required_cols:
        if col not in schedule_df.columns:
            raise ValueError(f"Missing column: {col}")
    # Sort by aircraft and time
    schedule_df = schedule_df.sort_values(['tail_number', 'sched_dep'])
    G = nx.DiGraph()
    # Add nodes
    for idx, row in schedule_df.iterrows():
        G.add_node(idx, **row)
    # Build edges for tight turnarounds
    for tail, group in schedule_df.groupby('tail_number'):
        flights = group.sort_values('sched_dep')
        for i in range(len(flights)-1):
            a = flights.iloc[i]
            b = flights.iloc[i+1]
            if pd.notna(a['actual_arr']) and pd.notna(b['sched_dep']):
                turnaround = (b['sched_dep'] - a['actual_arr']).total_seconds() / 60
                if turnaround < min_turn_required:
                    G.add_edge(a.name, b.name, turnaround=turnaround)
    # Find top 10 critical flights by out-degree
    out_degrees = G.out_degree()
    critical_flights = sorted(out_degrees, key=lambda x: x[1], reverse=True)[:10]
    return G, critical_flights

def simulate_cascading_delay(schedule_df, flight_idx, delay_minutes, min_turn_required=45):
    """
    Simulate a delay for a selected flight and propagate through aircraft chains.
    Returns: affected_flights (set of indices), affected_airports (set), affected_aircraft (set)
    """
    import copy
    df = schedule_df.copy()
    # Build graph
    G, _ = build_cascading_graph(df, min_turn_required)
    # Apply delay to selected flight
    df.at[flight_idx, 'actual_arr'] = df.at[flight_idx, 'actual_arr'] + pd.Timedelta(minutes=delay_minutes)
    affected_flights = set()
    affected_airports = set()
    affected_aircraft = set()
    # BFS to propagate delay
    queue = [flight_idx]
    while queue:
        current = queue.pop(0)
        affected_flights.add(current)
        affected_aircraft.add(df.at[current, 'tail_number'])
        affected_airports.add(df.at[current, 'To'] if 'To' in df.columns else None)
        for succ in G.successors(current):
            prev_arr = df.at[current, 'actual_arr']
            next_dep = df.at[succ, 'sched_dep']
            turnaround = (next_dep - prev_arr).total_seconds() / 60
            if turnaround < min_turn_required:
                # Propagate delay: update actual_arr for next flight
                delay_needed = min_turn_required - turnaround
                df.at[succ, 'actual_arr'] = prev_arr + pd.Timedelta(minutes=min_turn_required)
                if succ not in affected_flights:
                    queue.append(succ)
    return affected_flights, affected_airports, affected_aircraft, df
    for idx, row in schedule_df.iterrows():
        G.add_node(idx, **row)
    # Build edges for tight turnarounds
    for tail, group in schedule_df.groupby('tail_number'):
        flights = group.sort_values('sched_dep')
        for i in range(len(flights)-1):
            a = flights.iloc[i]
            b = flights.iloc[i+1]
            if pd.notna(a['actual_arr']) and pd.notna(b['sched_dep']):
                turnaround = (b['sched_dep'] - a['actual_arr']).total_seconds() / 60
                if turnaround < min_turn_required:
                    G.add_edge(a.name, b.name, turnaround=turnaround)
    # Find top 10 critical flights by out-degree
    out_degrees = G.out_degree()
    critical_flights = sorted(out_degrees, key=lambda x: x[1], reverse=True)[:10]
    return G, critical_flights
