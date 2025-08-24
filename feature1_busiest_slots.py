import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytz


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

# --- Inspect columns ---
print('Columns:', df.columns)
print(df.head())



# --- Combine Date and Time columns to create full datetimes ---
import re
timezone = pytz.timezone('Asia/Kolkata')

def combine_date_time(row, date_col, time_col):
    date_val = row[date_col]
    time_val = row[time_col]
    if pd.isna(date_val) or pd.isna(time_val):
        return pd.NaT
    # If time is already a datetime.time or string
    try:
        # If time is string like '06:00:00'
        t = pd.to_datetime(str(time_val)).time()
        dt = pd.Timestamp(date_val).replace(hour=t.hour, minute=t.minute, second=t.second)
        return timezone.localize(dt)
    except Exception:
        return pd.NaT

# Parse Date column as date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

# Scheduled Departure/Arrival
if 'STD' in df.columns and 'Date' in df.columns:
    df['sched_dep'] = df.apply(lambda row: combine_date_time(row, 'Date', 'STD'), axis=1)
if 'STA' in df.columns and 'Date' in df.columns:
    df['sched_arr'] = df.apply(lambda row: combine_date_time(row, 'Date', 'STA'), axis=1)
# Actual Departure
if 'ATD' in df.columns and 'Date' in df.columns:
    df['actual_dep'] = df.apply(lambda row: combine_date_time(row, 'Date', 'ATD'), axis=1)
# Actual Arrival (ATA: e.g., 'Landed 8:01 AM')
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

# --- Slotting: 15-min and 1-hr slots ---
if 'sched_dep' in df.columns:
    if df['sched_dep'].notna().sum() == 0:
        print('Warning: No valid scheduled departure datetimes found.')
    else:
        df['dep_slot_15min'] = df['sched_dep'].dt.floor('15min')
        df['dep_slot_1hr'] = df['sched_dep'].dt.floor('1h')
if 'sched_arr' in df.columns:
    if df['sched_arr'].notna().sum() == 0:
        print('Warning: No valid scheduled arrival datetimes found.')
    else:
        df['arr_slot_15min'] = df['sched_arr'].dt.floor('15min')
        df['arr_slot_1hr'] = df['sched_arr'].dt.floor('1h')


# --- Count movements per slot ---
movements_15min = pd.DataFrame()
movements_1hr = pd.DataFrame()
if 'dep_slot_15min' in df.columns or 'arr_slot_15min' in df.columns:
    dep_15 = df[['dep_slot_15min']].rename(columns={'dep_slot_15min': 'slot'}).dropna() if 'dep_slot_15min' in df.columns else pd.DataFrame()
    arr_15 = df[['arr_slot_15min']].rename(columns={'arr_slot_15min': 'slot'}).dropna() if 'arr_slot_15min' in df.columns else pd.DataFrame()
    if not dep_15.empty or not arr_15.empty:
        movements_15min = pd.concat([dep_15, arr_15])
        movements_15min = movements_15min.groupby('slot').size().reset_index(name='movements')
if 'dep_slot_1hr' in df.columns or 'arr_slot_1hr' in df.columns:
    dep_1h = df[['dep_slot_1hr']].rename(columns={'dep_slot_1hr': 'slot'}).dropna() if 'dep_slot_1hr' in df.columns else pd.DataFrame()
    arr_1h = df[['arr_slot_1hr']].rename(columns={'arr_slot_1hr': 'slot'}).dropna() if 'arr_slot_1hr' in df.columns else pd.DataFrame()
    if not dep_1h.empty or not arr_1h.empty:
        movements_1hr = pd.concat([dep_1h, arr_1h])
        movements_1hr = movements_1hr.groupby('slot').size().reset_index(name='movements')

# --- Runway capacity ---
RUNWAY_CAPACITY_HOURLY = 46
RUNWAY_CAPACITY_15MIN = RUNWAY_CAPACITY_HOURLY // 4

# --- Flag overloads ---
movements_15min['overload'] = movements_15min['movements'] > RUNWAY_CAPACITY_15MIN
movements_1hr['overload'] = movements_1hr['movements'] > RUNWAY_CAPACITY_HOURLY


# --- Top 10 busiest slots ---
if not movements_15min.empty:
    top_15min = movements_15min.sort_values('movements', ascending=False).head(10)
    print('\nTop 10 busiest 15-min slots:')
    print(top_15min)
else:
    print('\nNo valid 15-min slot data available.')
if not movements_1hr.empty:
    top_1hr = movements_1hr.sort_values('movements', ascending=False).head(10)
    print('\nTop 10 busiest 1-hr slots:')
    print(top_1hr)
else:
    print('\nNo valid 1-hr slot data available.')


# --- Heatmap of flight density by hour of day ---
if 'sched_dep' in df.columns and df['sched_dep'].notna().sum() > 0:
    df['hour'] = df['sched_dep'].dt.hour
    hourly_counts = df.groupby('hour').size()
    if not hourly_counts.empty:
        plt.figure(figsize=(10,4))
        sns.heatmap(hourly_counts.values.reshape(-1,1), annot=True, fmt='d', cmap='Reds', yticklabels=hourly_counts.index)
        plt.title('Flight Density by Hour of Day (Departures)')
        plt.xlabel('')
        plt.ylabel('Hour of Day')
        plt.tight_layout()
        plt.savefig('flight_density_heatmap.png')
        plt.show()
    else:
        print('No valid hourly departure data for heatmap.')
else:
    print('No valid scheduled departure data for heatmap.')

# --- Recommendation ---
if 'top_1hr' in locals() and not top_1hr.empty:
    busy_periods = top_1hr[top_1hr['overload']]['slot'].dt.strftime('%H:%M').tolist()
    if busy_periods:
        print(f"Recommendation: Avoid adding flights during these hours: {', '.join(busy_periods)}")
    else:
        print("No overloads detected in any slot.")
else:
    print("No recommendation possible due to lack of valid slot data.")
