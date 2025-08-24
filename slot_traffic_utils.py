import pandas as pd

def calculate_slot_traffic(schedule_df, airport, sched_dep_time):
    """
    Calculate the number of flights scheduled to depart from a given airport
    in the same 15-minute slot as sched_dep_time.
    schedule_df: DataFrame with at least columns ['From', 'sched_dep']
    airport: str, e.g., 'BOM'
    sched_dep_time: pd.Timestamp
    Returns: int (slot traffic count)
    """
    if pd.isna(sched_dep_time):
        return 1  # If time is missing, assume only this flight
    slot_start = sched_dep_time.floor('15min')
    slot_end = slot_start + pd.Timedelta(minutes=15)
    mask = (
        (schedule_df['From'] == airport) &
        (schedule_df['sched_dep'] >= slot_start) &
        (schedule_df['sched_dep'] < slot_end)
    )
    return mask.sum() + 1  # +1 to include the new flight itself

# Example usage:
# schedule = pd.read_excel('flight.xlsx')
# schedule['sched_dep'] = pd.to_datetime(schedule['sched_dep'])
# slot_traffic = calculate_slot_traffic(schedule, 'BOM', pd.Timestamp('2025-08-23 09:00'))
# print('Slot traffic:', slot_traffic)
