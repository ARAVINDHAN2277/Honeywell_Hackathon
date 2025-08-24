import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import pickle
import json
import matplotlib.pyplot as plt

print('Loading data...')
df = pd.read_excel('flight.xlsx')

def combine_date_time(row, date_col, time_col):
    date_val = row[date_col]
    time_val = row[time_col]
    if pd.isna(date_val) or pd.isna(time_val):
        return pd.NaT
    try:
        if isinstance(time_val, str) and ('AM' in time_val or 'PM' in time_val):
            t = pd.to_datetime(time_val).time()
        else:
            t = pd.to_datetime(str(time_val)).time()
        dt = pd.to_datetime(date_val).replace(hour=t.hour, minute=t.minute, second=t.second)
        return dt
    except Exception:
        return pd.NaT

df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
if 'STD' in df.columns and 'Date' in df.columns:
    df['sched_dep'] = df.apply(lambda row: combine_date_time(row, 'Date', 'STD'), axis=1)
if 'ATD' in df.columns and 'Date' in df.columns:
    df['actual_dep'] = df.apply(lambda row: combine_date_time(row, 'Date', 'ATD'), axis=1)
if 'STA' in df.columns and 'Date' in df.columns:
    df['sched_arr'] = df.apply(lambda row: combine_date_time(row, 'Date', 'STA'), axis=1)
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

print('Engineering features...')
df['dep_delay'] = (df['actual_dep'] - df['sched_dep']).dt.total_seconds() / 60
df['arr_delay'] = (df['actual_arr'] - df['sched_arr']).dt.total_seconds() / 60
df['delayed'] = (df['arr_delay'] > 15).astype(int)
df['hour'] = df['sched_dep'].dt.hour
df['dayofweek'] = pd.to_datetime(df['sched_dep']).dt.dayofweek
if 'From' in df.columns and 'To' in df.columns:
    df['route'] = df['From'].astype(str) + '-' + df['To'].astype(str)
else:
    df['route'] = 'Unknown'
if 'Aircraft' in df.columns:
    df['aircraft_type'] = df['Aircraft'].astype(str)
else:
    df['aircraft_type'] = 'Unknown'
df['dep_slot_15min'] = df['sched_dep'].dt.floor('15min')
slot_traffic = df.groupby('dep_slot_15min').size().rename('slot_traffic')
df = df.merge(slot_traffic, left_on='dep_slot_15min', right_index=True, how='left')
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
    def get_weather_features(dep_time):
        if pd.isna(dep_time):
            return [np.nan, np.nan, np.nan, np.nan]
        idx = np.argmin(np.abs((weather_df['weather_time'] - dep_time).dt.total_seconds()))
        row = weather_df.iloc[idx]
        return [row['temperature_2m'], row['windspeed_10m'], row['precipitation'], row['cloudcover']]
    weather_feats = df['sched_dep'].apply(get_weather_features)
    weather_feats_df = pd.DataFrame(weather_feats.tolist(), index=df.index, columns=['temperature_2m', 'windspeed_10m', 'precipitation', 'cloudcover'])
    df = pd.concat([df, weather_feats_df], axis=1)
    print('Weather features merged. Sample:')
    print(df[['sched_dep', 'temperature_2m', 'windspeed_10m', 'precipitation', 'cloudcover']].head())
except Exception as e:
    print(f'Could not add weather features: {e}')
    for col in ['temperature_2m', 'windspeed_10m', 'precipitation', 'cloudcover']:
        if col not in df.columns:
            df[col] = np.nan

features = ['hour', 'dayofweek', 'route', 'aircraft_type', 'slot_traffic', 'temperature_2m', 'windspeed_10m', 'precipitation', 'cloudcover', 'dep_delay']
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns in DataFrame: {missing_cols}")
    print("Available columns:", df.columns.tolist())
    raise ValueError(f"Missing columns for modeling: {missing_cols}")
X = pd.get_dummies(df[features], drop_first=True)
y = df['delayed']
mask = (~X.isnull().any(axis=1)) & (~y.isnull())
X = X[mask]
y = y[mask]

print('Training ensemble model (Random Forest + XGBoost)...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=(y_train == 0).sum()/(y_train == 1).sum())
ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb_clf)], voting='soft')
ensemble.fit(X_train, y_train)

print('\n--- Model Performance (Ensemble) ---')
y_prob = ensemble.predict_proba(X_test)[:,1]
threshold = 0.2  # Lowered threshold for more sensitive delay prediction
y_pred = (y_prob > threshold).astype(int)
print(f'Using threshold: {threshold}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.2f}')
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}')
print(classification_report(y_test, y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

# Save the ensemble model and columns for UI prediction
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('ensemble_model_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print('\nPipeline complete. Ensemble model run!')
