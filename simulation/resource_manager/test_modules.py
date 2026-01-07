import os
import pandas as pd
import random
from datetime import datetime, timedelta
# Import your specific Resource Manager class
from simulation.resource_manager.resource_manager import AdvancedResourceManager

TEST_CSV = "test_advanced_log.csv"


def create_advanced_dummy_log():
    """
    Creates a dummy log to test Role Discovery and Profiles.
    UPDATED: Generates a full week (Mon-Fri) so the Miner learns standard work days.
    """
    data = {
        'case_id': [],
        'activity': [],
        'timestamp': [],
        'org:resource': []
    }

    def add_event(case, act, time_str, res):
        data['case_id'].append(case)
        data['activity'].append(act)
        data['timestamp'].append(time_str)
        data['org:resource'].append(res)

    # Generate data for a full work week (Mon-Fri)
    # Jan 6 (Mon) to Jan 10 (Fri)
    base_date = datetime(2025, 1, 6)
    days = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)]

    for day in days:
        # User 1 & 2: The "Clerks" (Same tasks, roughly same times)
        add_event('c1', 'Task A', f'{day} 09:30:00+00:00', 'User_1')
        add_event('c1', 'Task B', f'{day} 16:30:00+00:00', 'User_1')

        add_event('c2', 'Task A', f'{day} 09:15:00+00:00', 'User_2')
        add_event('c2', 'Task B', f'{day} 16:45:00+00:00', 'User_2')

        # User 3: The "Manager" (Different Task)
        add_event('c3', 'Task C', f'{day} 10:30:00+00:00', 'User_3')

    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)
    print(f"[Setup] Advanced Dummy-Log created: {TEST_CSV}")


def cleanup():
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)
        print("[Cleanup] Test file removed.")


def test_advanced_features():
    print("\n--- TEST: Advanced Resource Manager (1.5, 1.6, 1.7) ---")

    # 1. Initialize (Start on a Monday)
    sim_start = datetime(2025, 1, 6, 8, 0, 0)
    rm = AdvancedResourceManager(sim_start)

    # 2. Mining & Role Discovery (Task 1.6)
    rm.load_log_and_mine_profiles(TEST_CSV)

    print(f"\n[Check 1.6] Role Discovery (Clustering):")
    users_clustered = False
    for role, members in rm.roles.items():
        if 'User_1' in members and 'User_2' in members:
            users_clustered = True
            print(f"  -> SUCCESS: User_1 and User_2 grouped in {role}: {members}")
            break

    if not users_clustered:
        print("  -> FAIL: User_1 and User_2 were not grouped together (Check K-Means logic).")

    # 3. Availability (Shift Check - Task 1.5)
    print("\n[Check 1.5] Shift Constraints:")
    is_working = rm.check_availability("User_1", datetime(2025, 1, 6, 10, 0, 0))  # Mon 10am
    is_sleeping = rm.check_availability("User_1", datetime(2025, 1, 6, 4, 0, 0))  # Mon 4am

    if is_working and not is_sleeping:
        print("  -> SUCCESS: Shift logic works (Active day, Inactive night).")
    else:
        print(f"  -> FAIL: Working={is_working}, Sleeping={is_sleeping}")

    # 4. Stochastic Tests (Illness/Interruptions - Task 1.5 Advanced)
    print("\n[Check 1.5] Stochastic Logic:")

    # TEST A: ILLNESS (Check 100 different DAYS)
    # We ignore Micro-Interruptions (5%) here by setting a threshold.
    # Total Unavailability = Sickness (2%) + Micro (5%) + Random Noise
    unavailable_count = 0
    workdays_checked = 0

    for i in range(150):
        test_date = sim_start + timedelta(days=i)

        # Only check Mon-Fri (0-4) because the Miner now knows these are the only workdays
        if test_date.weekday() <= 4:
            workdays_checked += 1
            # Check availability at 10:00 AM
            if not rm.check_availability("User_1", test_date.replace(hour=10)):
                unavailable_count += 1

    # We expect roughly 7% total rejection (2% sick + 5% micro-interruption)
    # 7% of 100 days is ~7 days.
    print(f"  Simulated {workdays_checked} workdays. Unavailable days: {unavailable_count}")

    if 0 < unavailable_count < (workdays_checked * 0.15):
        print("  -> SUCCESS: Stochastic logic is working reasonably (Sick + Interruptions).")
    elif unavailable_count == 0:
        print("  -> NOTE: No unavailability found (Random chance).")
    else:
        print(f"  -> WARNING: Rate is high ({unavailable_count}/{workdays_checked}). Check Profile Work Days.")

    # TEST B: MICRO-INTERRUPTIONS
    # Find a healthy day first (one where check returns True at least once)
    healthy_date = None
    for i in range(10):
        d = sim_start + timedelta(days=i)
        if d.weekday() == 0 and rm.check_availability("User_1", d.replace(hour=10)):
            healthy_date = d.replace(hour=10)
            break

    if healthy_date:
        interruptions = 0
        for _ in range(200):
            if not rm.check_availability("User_1", healthy_date):
                interruptions += 1

        print(f"  Simulated 200 requests on a healthy day. Interruptions: {interruptions}")
        if 0 < interruptions < 40:
            print("  -> SUCCESS: Micro-interruptions occurring (approx 5% rate).")
        else:
            print("  -> WARNING: Interruption rate is odd (0 or too high).")
    else:
        print("  -> SKIP: Could not find a healthy day.")


if __name__ == "__main__":
    try:
        create_advanced_dummy_log()
        test_advanced_features()
    finally:
        cleanup()