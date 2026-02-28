import os
import pandas as pd
import numpy as np
import pm4py
from datetime import datetime
from resource_manager import AdvancedResourceManager, RandomPlanner, GreedyPlanner, AdvancedOptimizationPlanner


def extract_tasks_from_log(df, num_cases=100):
    case_col = 'case:concept:name' if 'case:concept:name' in df.columns else 'case_id'
    unique_cases = df[case_col].unique()[:num_cases]
    test_df = df[df[case_col].isin(unique_cases)].copy()
    test_df['time:timestamp'] = pd.to_datetime(test_df['time:timestamp'], utc=True)
    test_df = test_df.sort_values(by=[case_col, 'time:timestamp'])
    test_df['duration'] = (test_df.groupby(case_col)['time:timestamp'].shift(-1) - test_df[
        'time:timestamp']).dt.total_seconds().fillna(900)
    start = test_df['time:timestamp'].min()
    tasks = []
    for _, row in test_df.iterrows():
        tasks.append({
            'activity': row['concept:name'], 'arrival': (row['time:timestamp'] - start).total_seconds(),
            'duration': max(60, row['duration']), 'priority': row['case:RequestedAmount'] > 20000,
            'amount': row['case:RequestedAmount']
        })
    return sorted(tasks, key=lambda x: x['arrival'])


class FullDataEvaluator:
    def __init__(self, manager):
        self.manager = manager

    def run_full_evaluation(self, tasks):
        for name, planner in [("Random", RandomPlanner()), ("Greedy", GreedyPlanner()),
                              ("Advanced", AdvancedOptimizationPlanner())]:
            self.manager.strategy, self.manager.busy_until, self.manager.daily_work_seconds, self.manager.last_activity = planner, {}, {}, {}
            results = []
            for i, t in enumerate(tasks):
                self.manager.update_predictions(
                    [(future['activity'], future['arrival'], future['priority']) for future in tasks[i + 1:i + 6]])
                wait, res = 0, None
                while res is None and wait < 86400:
                    res = self.manager.request_resource(t['activity'], t['arrival'] + wait, t['duration'],
                                                        case_amount=t['amount'])
                    if res is None: wait += 60
                results.append({'wait': wait, 'res': res})
            print(
                f"DONE {name}: Avg Wait: {np.mean([r['wait'] for r in results]) / 60:.2f}m | Offload: {(len([r for r in results if r['res'] in self.manager.system_resources]) / len(results)) * 100:.1f}%")


def main():
    parquet_path = "../../data/bpi-chall.parquet"
    if not os.path.exists(parquet_path):
        df = pm4py.convert_to_dataframe(pm4py.read_xes("../../data/bpi-chall.xes"))
        df.to_parquet(parquet_path)
    else:
        df = pd.read_parquet(parquet_path)

    tasks = extract_tasks_from_log(df, num_cases=5000)
    manager = AdvancedResourceManager(datetime(2025, 1, 1), RandomPlanner())
    manager.load_log_and_mine_profiles(df)
    FullDataEvaluator(manager).run_full_evaluation(tasks)


if __name__ == "__main__": main()