import os
import pandas as pd
import random
from datetime import datetime, timedelta
# UPDATE: Import the Advanced Manager
from simulation.resource_manager.resource_manager import AdvancedResourceManager

TEST_CSV = "test_advanced_log.csv"


def create_advanced_dummy_log():
    """
    Creates a dummy log to test Role Discovery and Profiles.
    - User_1 and User_2 do the SAME tasks (Task A, Task B) -> Should be clustered together (Role 1).
    - User_3 does distinct tasks (Task C) -> Should be separate (Role 2).
    - Timestamps set to establish a 9-5 working profile.
    """
    data = {
        'case_id': [],
        'activity': [],
        'timestamp': [],
        'resource': []
    }

    # Helper to add row
    def add_event(case, act, time_str, res):
        data['case_id'].append(case)
        data['activity'].append(act)
        data['timestamp'].append(time_str)
        data['resource'].append(res)

    # Generate data for 3 days to establish profiles
    # User 1 & 2: Working 09:00 - 17:00
    # User 3: Working 10:00 - 18:00
    days = ['2025-01-05', '2025-01-06', '2025-01-07']  # Sun, Mon, Tue

    for day in days:
        # User 1 (Task A, Task B)
        add_event('c1', 'Task A', f'{day} 09:30:00+00:00', 'User_1')
        add_event('c1', 'Task B', f'{day} 16:30:00+00:00', 'User_1')

        # User 2 (Task A, Task B) - Similar behavior to User 1
        add_event('c2', 'Task A', f'{day} 09:15:00+00:00', 'User_2')
        add_event('c2', 'Task B', f'{day} 16:45:00+00:00', 'User_2')

        # User 3 (Task C) - Different behavior
        add_event('c3', 'Task C', f'{day} 10:30:00+00:00', 'User_3')
        add_event('c3', 'Task C', f'{day} 17:30:00+00:00', 'User_3')

    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)
    print(f"[Setup] Advanced Dummy-Log created: {TEST_CSV}")


def cleanup():
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)
        print("[Cleanup] Test file removed.")


def test_advanced_features():
    print("\n--- Testing ADVANCED Resource Manager (1.5, 1.6, 1.7) ---")

    # 1. Initialize (Start on a Monday)
    sim_start = datetime(2025, 1, 6, 8, 0, 0)  # Mon 8:00
    rm = AdvancedResourceManager(sim_start)

    # 2. Test Mining & Role Discovery (Task 1.6 Advanced)
    rm.load_log_and_mine_profiles(TEST_CSV)

    print("\n[Check 1.6] Role Discovery via Clustering:")
    print(f"  Discovered Roles: {len(rm.roles)}")
    for role, members in rm.roles.items():
        print(f"  - {role}: {members}")

    # Validation logic: User_1 and User_2 should ideally be in the same role group
    # (Since they perform exactly the same set of activities: {Task A, Task B})

    print("\n[Check 1.5] Profile Mining (Shifts):")
    if 'User_1' in rm.profiles:
        p = rm.profiles['User_1']
        print(f"  User_1 Profile: {p.start_hour}:00 to {p.end_hour}:00")
        # Expectation: Start around 9, End around 16/17
        if p.start_hour <= 10 and p.end_hour >= 16:
            print("  -> SUCCESS: Working hours mined correctly.")
        else:
            print("  -> FAIL: Working hours look wrong.")
    else:
        print("  -> FAIL: User_1 profile not found.")

    # 3. Test Availability & Request (Task 1.5 Advanced)
    print("\n[Check 1.7] Request & Stochastic Availability:")

    # Case A: Request at 8:00 AM (Before shift)
    # User 1 starts at ~9:00. 8:00 should be too early.
    res = rm.request_resource('Task A', current_sim_time=0, duration=60)  # 0 = 8:00 AM
    if res is None:
        print("  [08:00 AM] Correctly returned None (Shift hasn't started).")
    else:
        print(f"  [08:00 AM] FAIL: Resource {res} assigned outside shift!")

    # Case B: Request at 10:00 AM (During shift)
    # 2 hours later = 7200 seconds
    res_valid = rm.request_resource('Task A', current_sim_time=7200, duration=60)
    if res_valid:
        print(f"  [10:00 AM] SUCCESS: Assigned {res_valid}.")
    else:
        print("  [10:00 AM] FAIL: No resource assigned (Check availability logic).")

    # Case C: Stochastic Check (Illness/Lunch)
    # We simulate 100 requests to see if we get *some* rejections due to the stochastic factors
    # (Illness 2%, Lunch 30% if time is 12-14)
    print("\n[Check Stochastic] Simulating 100 requests at 12:30 (Lunch time)...")
    rejections = 0
    lunch_seconds = 4.5 * 3600  # 12:30 PM

    for _ in range(100):
        # We assume simulation runs on different days or we reset busy_until
        rm.busy_until = {}  # Force free
        # Note: Illness is seeded by DAY, so it won't fluctuate in a tight loop on the same day.
        # But Lunch (random.random()) IS volatile per request in your code.
        if rm.request_resource('Task A', lunch_seconds, 60) is None:
            rejections += 1

    print(f"  Rejections at lunch time: {rejections}/100")
    if rejections > 0:
        print("  -> SUCCESS: Stochastic unavailability (Lunch/Interruptions) is working.")
    else:
        print("  -> WARNING: No stochastic rejections. Check probabilities.")


if __name__ == "__main__":
    try:
        create_advanced_dummy_log()
        test_advanced_features()
    finally:
        cleanup()