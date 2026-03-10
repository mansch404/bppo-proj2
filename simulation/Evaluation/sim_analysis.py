import pandas as pd

# Load the clean log from your best method's first run
df = pd.read_csv("eval_logs_1/Advanced_Optimizer_run_1_CLEAN.csv")

# Filter out the system/bots
humans_only = df[df['is_system'] == False]

# Sum the total service seconds per human
workload = humans_only.groupby('resource')['service_seconds'].sum().sort_values()

print("Least utilized employees (in seconds of work):")
print(workload.head(5))

