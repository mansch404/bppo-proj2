import pm4py
import pandas as pd

# 1. Load the real historical data
print("Loading event log...")
log = pm4py.read_xes(r"C:\Users\kickb\OneDrive\Escritorio\bppo-proj2\data\bpi-chall.xes")
df = pm4py.convert_to_dataframe(log)

# 2. Standardize timestamp and lifecycle columns
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
df['lifecycle:transition'] = df['lifecycle:transition'].str.lower()

# 3. Isolate 'start' and 'complete' events
starts = df[df['lifecycle:transition'] == 'start'].copy()
completes = df[df['lifecycle:transition'] == 'complete'].copy()

# 4. Sequence them to pair the exact start and complete of a specific task within a case
starts['seq'] = starts.groupby(['case:concept:name', 'concept:name']).cumcount()
completes['seq'] = completes.groupby(['case:concept:name', 'concept:name']).cumcount()

# 5. Merge to create a wide format (1 row = 1 task with start and end times)
merged = pd.merge(
    starts,
    completes,
    on=['case:concept:name', 'concept:name', 'seq'],
    suffixes=('_start', '_complete')
)

# 6. Calculate precise duration in seconds
merged['duration_seconds'] = (merged['time:timestamp_complete'] - merged['time:timestamp_start']).dt.total_seconds()

# Filter out any negative anomalies (sometimes present in messy real-world logs)
merged = merged[merged['duration_seconds'] >= 0]

# 7. Sum the total working time per resource (using the resource logged at completion)
workload_seconds = merged.groupby('org:resource_complete')['duration_seconds'].sum()

# 8. Filter out the system bot ('User_1')
human_workload = workload_seconds[workload_seconds.index != 'User_1']

# 9. Sort to find the least utilized
least_utilized = human_workload.sort_values()

print("\nLeast utilized human employees historically (by exact seconds of work):")
# Displaying the bottom 5. The first 2 are your targets to fire!
for res, secs in least_utilized.head(5).items():
    print(f"  {res}: {secs:.0f} seconds (approx {secs/3600:.1f} hours)")