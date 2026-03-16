import os

# CRITICAL: Must be at the very top
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from heuristic_evaluator import FullScaleEvaluator
from resource_manager import DeepRLPlanner


def train_agent(log_path_str, epochs=100):
    # --- PROPER PATH RESOLUTION ---
    # Find the directory where this script lives
    script_dir = Path(__file__).resolve().parent
    # Project root is two levels up from simulation/resource_manager/
    project_root = script_dir.parents[1]

    # Try the provided path relative to root
    absolute_log_path = project_root / log_path_str

    if not absolute_log_path.exists():
        # Fallback: check if it's just in the project root data folder
        absolute_log_path = project_root / "data" / "bpi-chall.xes"

    print(f"Checking path: {absolute_log_path}")
    if not absolute_log_path.exists():
        raise FileNotFoundError(f"Could not find XES file at {absolute_log_path}. "
                                f"Please ensure it is in the project root/data/ folder.")

    # 1. Initialize Evaluator
    evaluator = FullScaleEvaluator(str(absolute_log_path))

    all_res = []
    for role_res in evaluator._manager_template["roles"].values():
        all_res.extend(role_res)
    all_res = sorted(list(set(all_res)))

    # 2. Setup Planner and Optimizer (lr=5e-4 for stability)
    planner = DeepRLPlanner(all_resource_names=all_res, is_training=True)
    optimizer = optim.Adam(planner.model.parameters(), lr=5e-4)

    # 3. SPEED FIX: 50 cases per epoch for rapid feedback
    tasks = evaluator._build_tasks(num_cases=50)

    print(f"Starting Training: {len(all_res)} resources, {epochs} epochs...")

    all_rewards = []
    best_wait = float('inf')

    for epoch in range(epochs):
        # Run Simulation Episode
        metrics = evaluator.run_experiment(
            planner, "TrainingLoop", tasks, seed=epoch, progress_every=2000
        )

        # 4. LOGIC FIX: Advantage Reinforcement
        # 1. Get Metrics
        avg_wait = metrics.get("Avg Wait Time (min)", 100.0)
        timeout_rate = metrics.get("Timeout Rate (%)", 0.0)

        # 2. IMPROVED REWARD: Penalty for timeouts
        # We want to minimize wait, but timeouts are 10x more expensive.
        # This forces the agent to pick a resource instead of waiting forever.
        reward = -(avg_wait / 10.0) - (timeout_rate / 2.0)

        all_rewards.append(reward)

        # 3. Advantage Calculation (Moving Baseline)
        baseline = np.mean(all_rewards[-10:]) if len(all_rewards) > 1 else reward
        advantage = torch.tensor(reward - baseline, device=planner.device, dtype=torch.float)

        # 4. Standard Policy Update
        optimizer.zero_grad()
        policy_loss = []
        for log_prob in planner.saved_log_probs:
            policy_loss.append(-log_prob * advantage)

        if policy_loss:
            loss = torch.stack(policy_loss).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(planner.model.parameters(), max_norm=1.0)
            optimizer.step()

        planner.saved_log_probs = []  # Reset memory

        # Save the Best Model
        if avg_wait < best_wait:
            best_wait = avg_wait
            torch.save(planner.model.state_dict(), "rl_model_best.pt")
            print(f"--> Epoch {epoch}: New Best Wait {best_wait:.2f}m (Saved)")

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Wait: {avg_wait:.2f}m | Adv: {advantage:.4f}")

    print("Training complete. Best model is 'rl_model_best.pt'")


if __name__ == "__main__":
    train_agent(log_path_str="data/bpi-chall.xes")