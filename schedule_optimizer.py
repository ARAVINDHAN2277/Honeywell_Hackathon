from ortools.sat.python import cp_model
import pandas as pd

def optimize_schedule(schedule_df, runway_capacity=2, slot_minutes=15, shift_window=15, turnaround_min=60):
    """
    Optimize flight departures to minimize total predicted delay, subject to constraints.
    - Each flight can be shifted ±shift_window minutes
    - Runway capacity: max flights per slot
    - Turnaround: min time between flights for same tail_number
    Returns: DataFrame with optimized departure times
    """
    model = cp_model.CpModel()
    # Keep original index to merge results back later
    flights = schedule_df.copy().reset_index()  # creates column 'index' with original row index

    # Parse sched_dep safely and exclude rows without a valid scheduled departure
    flights['sched_dep'] = pd.to_datetime(flights['sched_dep'], errors='coerce')
    missing_count = flights['sched_dep'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} flights missing sched_dep; they will be excluded from optimization")
    flights_valid = flights[flights['sched_dep'].notna()].copy().reset_index(drop=True)
    n = len(flights_valid)
    if n == 0:
        print('No valid flights with sched_dep found for optimization.')
        return None

    # Convert scheduled departure to minutes since midnight
    flights_valid['dep_min'] = flights_valid['sched_dep'].dt.hour * 60 + flights_valid['sched_dep'].dt.minute

    # Decision variables: new departure time (in minutes)
    dep_vars = []
    for i in range(n):
        orig = int(flights_valid.at[i, 'dep_min'])
        var = model.NewIntVar(orig - shift_window, orig + shift_window, f'dep_{i}')
        dep_vars.append(var)

    # --- Runway capacity constraint ---
    min_dep = int(flights_valid['dep_min'].min()) - shift_window
    max_dep = int(flights_valid['dep_min'].max()) + shift_window
    for slot_start in range(min_dep, max_dep + 1, slot_minutes):
        in_slot = []
        for i in range(n):
            b = model.NewBoolVar(f'in_slot_{i}_{slot_start}')
            model.Add(dep_vars[i] >= slot_start).OnlyEnforceIf(b)
            model.Add(dep_vars[i] < slot_start + slot_minutes).OnlyEnforceIf(b)
            in_slot.append(b)
        model.Add(sum(in_slot) <= runway_capacity)

    # --- Turnaround constraint for same tail_number ---
    for tail in flights_valid['tail_number'].unique():
        idxs = flights_valid.index[flights_valid['tail_number'] == tail].tolist()
        idxs = sorted(idxs, key=lambda i: flights_valid.at[i, 'dep_min'])
        for i in range(len(idxs) - 1):
            model.Add(dep_vars[idxs[i+1]] >= dep_vars[idxs[i]] + turnaround_min)

    # --- Objective: minimize total positive delay ---
    delays = []
    for i in range(n):
        delay = model.NewIntVar(0, shift_window * 2, f'delay_{i}')
        model.AddMaxEquality(delay, [dep_vars[i] - int(flights_valid.at[i, 'dep_min']), 0])
        delays.append(delay)

    model.Minimize(sum(delays))

    # --- Solve ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        flights_valid['optimized_dep_min'] = [solver.Value(var) for var in dep_vars]
        # construct optimized_sched_dep from the date of sched_dep
        flights_valid['optimized_sched_dep'] = flights_valid['sched_dep'].dt.normalize() + pd.to_timedelta(flights_valid['optimized_dep_min'], unit='m')
        flights_valid['delay_minutes'] = flights_valid['optimized_dep_min'] - flights_valid['dep_min']
        # merge optimized results back to original dataframe
        result = flights.merge(flights_valid[['index', 'optimized_sched_dep', 'delay_minutes']], on='index', how='left')
        # ensure original order
        result = result.sort_values('index').reset_index(drop=True)
        return result
    else:
        return None
