
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import re

# Load both sheets and concatenate using correct names
sheet_names = ['6AM - 9AM', '9AM - 12PM']
dfs = []
for sheet in sheet_names:
    try:
        dfs.append(pd.read_excel('flight.xlsx', sheet_name=sheet))
    except Exception as e:
        print(f"Warning: Could not read sheet '{sheet}': {e}")
if dfs:
    df = pd.concat(dfs, ignore_index=True)
else:
    raise ValueError('No sheets could be loaded from flight.xlsx')

# --- Combine Date and Time columns to create full datetimes ---
timezone = pytz.timezone('Asia/Kolkata')
def combine_date_time(row, date_col, time_col):
    date_val = row[date_col]
    time_val = row[time_col]
    if pd.isna(date_val) or pd.isna(time_val):
        return pd.NaT
    try:
        t = pd.to_datetime(str(time_val)).time()
        dt = pd.Timestamp(date_val).replace(hour=t.hour, minute=t.minute, second=t.second)
        return timezone.localize(dt)
    except Exception:
        return pd.NaT
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
if 'STD' in df.columns and 'Date' in df.columns:
    df['sched_dep'] = df.apply(lambda row: combine_date_time(row, 'Date', 'STD'), axis=1)
if 'STA' in df.columns and 'Date' in df.columns:
    df['sched_arr'] = df.apply(lambda row: combine_date_time(row, 'Date', 'STA'), axis=1)
if 'ATD' in df.columns and 'Date' in df.columns:
    df['actual_dep'] = df.apply(lambda row: combine_date_time(row, 'Date', 'ATD'), axis=1)
def extract_time_from_ata(ata_str):
    if pd.isna(ata_str):
        return None
    match = re.search(r'(\d{1,2}:\d{2}\s*[APMapm]{2})', str(ata_str))
    if match:
        return match.group(1)
    match2 = re.search(r'(\d{1,2}:\d{2})', str(ata_str))
    if match2:
        return match2.group(1)
    return None
if 'ATA' in df.columns and 'Date' in df.columns:
    df['ATA_time'] = df['ATA'].apply(extract_time_from_ata)
    df['actual_arr'] = df.apply(lambda row: combine_date_time(row, 'Date', 'ATA_time'), axis=1)

# --- Feature 2: Find Best Time to Takeoff/Land ---
print("\n--- Feature 2: Find Best Time to Takeoff/Land ---")
df['dep_delay'] = (df['actual_dep'] - df['sched_dep']).dt.total_seconds() / 60
df['arr_delay'] = (df['actual_arr'] - df['sched_arr']).dt.total_seconds() / 60

recommendation = ""
summary = ""

if 'sched_dep' in df.columns and df['dep_delay'].notna().sum() > 0:
    df['dep_hour'] = df['sched_dep'].dt.hour
    dep_hourly_delay = df.groupby('dep_hour')['dep_delay'].mean()
    best_dep_hours = dep_hourly_delay.nsmallest(3)
    worst_dep_hours = dep_hourly_delay.nlargest(3)
    print("\nAverage departure delay by hour:")
    print(dep_hourly_delay)
    print("Best departure hours (lowest avg delay):")
    for h, v in best_dep_hours.items():
        if pd.isna(v):
            continue
        if v < 0:
            print(f"  {int(h):02d}:00 — on average {abs(v):.1f} minutes early")
        else:
            print(f"  {int(h):02d}:00 — on average {v:.1f} minutes late")
    # Print worst 3 hours
    worst_hours_str = ', '.join(f"{int(h):02d}:00" for h in worst_dep_hours.index)
    avoid_str = f"Avoid departures at: {worst_hours_str}."
    recommendation += avoid_str
    print(f"\nRecommendation: {avoid_str}")
    # Best 3 hour summary (professional wording)
    best_window = ' '.join([
        f"Flights departing at {int(h):02d}:00 are on average {abs(v):.1f} minutes early."
        if v < 0 else
        f"Flights departing at {int(h):02d}:00 are on average {v:.1f} minutes late."
        for h, v in best_dep_hours.items() if not pd.isna(v)
    ])
    summary += best_window
    print(f"\nSummary: {best_window}")
    plt.figure(figsize=(10,4))
    dep_hourly_delay.plot(marker='o')
    plt.title('Average Departure Delay by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Avg Departure Delay (min)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_dep_delay_by_hour.png')
    plt.show()
else:
    print('No valid scheduled/actual departure data for delay analysis.')

if 'sched_arr' in df.columns and df['arr_delay'].notna().sum() > 0:
    df['arr_hour'] = df['sched_arr'].dt.hour
    arr_hourly_delay = df.groupby('arr_hour')['arr_delay'].mean()
    best_arr_hours = arr_hourly_delay.nsmallest(3)
    best_arr_hours = arr_hourly_delay.nsmallest(3)
    worst_arr_hours = arr_hourly_delay.nlargest(3)
    print("\nAverage arrival delay by hour:")
    print(arr_hourly_delay)
    print("Best arrival hours (lowest avg delay):")
    for h, v in best_arr_hours.items():
        if pd.isna(v):
            continue
        if v < 0:
            print(f"  {int(h):02d}:00 — on average {abs(v):.1f} minutes early")
        else:
            print(f"  {int(h):02d}:00 — on average {v:.1f} minutes late")
    # Print worst 3 hours
    worst_hours_str = ', '.join(f"{int(h):02d}:00" for h in worst_arr_hours.index)
    avoid_str = f"Avoid arrivals at: {worst_hours_str}."
    recommendation += " " + avoid_str
    print(f"\nRecommendation: {avoid_str}")
    # Best 3 hour summary (professional wording)
    best_window = ' '.join([
        f"Flights arriving at {int(h):02d}:00 are on average {abs(v):.1f} minutes early."
        if v < 0 else
        f"Flights arriving at {int(h):02d}:00 are on average {v:.1f} minutes late."
        for h, v in best_arr_hours.items() if not pd.isna(v)
    ])
    summary += " " + best_window
    print(f"\nSummary: {best_window}")
    plt.figure(figsize=(10,4))
    arr_hourly_delay.plot(marker='o', color='green')
    plt.title('Average Arrival Delay by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Avg Arrival Delay (min)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('avg_arr_delay_by_hour.png')
    plt.show()
else:
    print('No valid scheduled/actual arrival data for delay analysis.')

print(f"\n--- DELIVERABLE ---\n{recommendation.strip()}\n{summary.strip()}")
