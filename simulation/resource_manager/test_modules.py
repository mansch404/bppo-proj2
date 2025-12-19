import os
import pandas as pd
from datetime import datetime, timedelta
from simulation.resource_manager.resource_manager import ResourceManager

# 1. Dummy-Daten erstellen
TEST_CSV = "test_log.csv"


def create_dummy_log():
    data = {
        'case_id': ['c1', 'c1', 'c2', 'c2'],
        'activity': ['Task A', 'Task A', 'Task A', 'Task A'],
        # Task A dauert hier immer ca. 60 Sekunden (10:00:00 bis 10:01:00)
        'timestamp': [
            '2025-01-01 10:00:00', '2025-01-01 10:01:00',
            '2025-01-01 11:00:00', '2025-01-01 11:01:00'
        ],
        'lifecycle': ['start', 'complete', 'start', 'complete'],
        'resource': ['User_1', 'User_1', 'User_2', 'User_2']
    }
    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)
    print(f"[Setup] Dummy-Log erstellt: {TEST_CSV}")


def cleanup():
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)
        print("[Cleanup] Test-Datei gelöscht.")


# --- TEST TASK 1.5, 1.6, 1.7 (RESOURCE MANAGER) ---
def test_resource_manager():
    print("\n--- Teste Task 1.5, 1.6, 1.7 (Resource Manager) ---")

    # Startzeit: Ein Montag um 08:00 Uhr
    sim_start = datetime(2026, 1, 5, 8, 0, 0)
    rm = ResourceManager(sim_start)

    # 1. Teste Loading (Task 1.6)
    rm.load_permissions_from_log(TEST_CSV)
    if 'Task A' in rm.permissions and 'User_1' in rm.permissions['Task A']:
        print("Task 1.6 (Permissions): User_1 wurde für Task A erkannt.")
    else:
        print("Task 1.6: Permissions nicht korrekt geladen.")

    # 2. Teste Schichtplan / Availability (Task 1.5)
    # Montag 10:00 Uhr -> Sollte OK sein
    monday_morning = sim_start + timedelta(hours=2)
    if rm.is_on_shift(monday_morning):
        print("Task 1.5 (Interval): Montag 10:00 ist Arbeitszeit.")
    else:
        print("Task 1.5: Montag 10:00 sollte Arbeitszeit sein!")

    # Sonntag 10:00 Uhr -> Sollte False sein
    sunday = sim_start + timedelta(days=6)
    if not rm.is_on_shift(sunday):
        print("Task 1.5 (Interval): Sonntag ist KEINE Arbeitszeit.")
    else:
        print("Task 1.5: Sonntag wurde fälschlicherweise als Arbeitszeit erkannt.")

    # 3. Teste Allocation (Task 1.7)
    print("\n--- Allocation Test ---")
    # Wir simulieren eine Anfrage bei Minute 0 (Montag 8 Uhr)
    sim_time_seconds = 0
    duration = 60  # 1 Minute

    # Erste Anfrage: Sollte klappen
    res1 = rm.request_resource('Task A', sim_time_seconds, duration)
    print(f"Erste Anfrage Ergebnis: {res1}")

    if res1:
        print("Task 1.7: Ressource erfolgreich zugewiesen.")
    else:
        print("Task 1.7: Zuweisung fehlgeschlagen (obwohl alle frei sein sollten).")

    # Zweite Anfrage SOFORT danach: Sollte eine ANDERE Ressource sein oder None (wenn nur 1 da wäre)
    # Unser Log hat User_1 und User_2. Einer ist jetzt beschäftigt.
    res2 = rm.request_resource('Task A', sim_time_seconds, duration)
    print(f"Zweite Anfrage Ergebnis: {res2}")

    if res2 and res2 != res1:
        print("Task 1.7: Zweite Ressource zugewiesen (User_1 und User_2 sind beide da).")
    elif res2 is None:
        print("Task 1.7: Keine zweite Ressource frei (Check deine Daten).")
    else:
        print("Task 1.7: Gleiche Ressource doppelt vergeben! (Fehler im Locking)")


if __name__ == "__main__":
    try:
        create_dummy_log()
        test_resource_manager()
    finally:
        cleanup()