import random
from typing import List, Set, Dict


class ProcessInstance:
    def __init__(self, case_id: str, start_time: float):
        self.case_id = case_id
        self.start_time = start_time
        self.status = "ACTIVE"

        # current_activities speichert, wo wir uns gerade befinden (Tokens)
        self.current_activities: Set[str] = {"Start"}

        # history speichert alles, was schon erledigt wurde
        self.history: List[str] = []

        # Zähler für Aktivitäten (hilfreich für Statistiken oder Debugging)
        self.activity_counts: Dict[str, int] = {}

        # FLAGS für Prozess-Steuerung
        # 1. Flag für den AND-Join (Parallel Gateway im ersten Teil)
        self.parallel_path_completed = {
            "A_Concept": False,
            "W_Complete application": False
        }
        # 2. Flag um zu unterscheiden, ob wir VOR oder NACH dem Angebot sind
        # Das löst dein Problem mit dem doppelten "W_Complete application"
        self.phase_after_offer = False

    def update_state(self, completed_activity: str) -> List[str]:
        # 1. Token entfernen
        if completed_activity in self.current_activities:
            self.current_activities.remove(completed_activity)

        # 2. Historie und Counter updaten
        self.history.append(completed_activity)
        self.activity_counts[completed_activity] = self.activity_counts.get(completed_activity, 0) + 1

        next_activities = []

        # --- ANFANG ---
        if completed_activity == "Start":
            next_activities = ["A_Create Application"]

        elif completed_activity == "A_Create Application":
            # Entscheidung: Skip to Parallel (Bypass) oder Normal
            # Wahrscheinlichkeit aus Daten!
            if random.random() < 0.2:
                # SKIP: Direkt zum Parallel Gateway
                next_activities = ["A_Concept", "W_Complete application"]
                self._reset_parallel_flags()
            else:
                next_activities = ["A_Submitted"]

        elif completed_activity == "A_Submitted":
            next_activities = ["W_Handle leads"]

        elif completed_activity == "W_Handle leads":
            # Entscheidung: Loop (Fehler) oder Weiter
            if random.random() < 0.3:  # Loop
                next_activities = ["W_Handle leads"]
            else:
                # WEITER zum Parallel Gateway
                next_activities = ["A_Concept", "W_Complete application"]
                self._reset_parallel_flags()

        # --- PARALLELER TEIL (Oben) ---
        elif completed_activity == "A_Concept":
            self.parallel_path_completed["A_Concept"] = True
            if self.parallel_path_completed["W_Complete application"]:
                next_activities = ["A_Accepted"]

        # --- W_COMPLETE APPLICATION (Das Problemkind) ---
        elif completed_activity == "W_Complete application":

            # HIER IST DEINE LOGIK MIT DEM FLAG / DER BEDINGUNG:
            if self.phase_after_offer == True:
                # Fall 2: Wir sind NACH dem Angebot (Sequenz)
                # Es geht weiter zum Call
                next_activities = ["W_Call after offers"]
            else:
                # Fall 1: Wir sind im PARALLELEN Teil (vor dem Angebot)

                # Check auf Loop im parallelen Teil
                if random.random() < 0.1:
                    next_activities = ["W_Complete application"]
                else:
                    self.parallel_path_completed["W_Complete application"] = True
                    # Check AND Join
                    if self.parallel_path_completed["A_Concept"]:
                        next_activities = ["A_Accepted"]

        # --- MITTELTEIL ---
        elif completed_activity == "A_Accepted":
            next_activities = ["O_Create Offer"]

        elif completed_activity == "O_Create Offer":
            next_activities = ["O_Created"]

        elif completed_activity == "O_Created":
            next_activities = ["O_Sent (mail and online)"]

        elif completed_activity == "O_Sent (mail and online)":
            # WICHTIG: Hier setzen wir das Flag, weil wir diesen Punkt passiert haben!
            self.phase_after_offer = True

            # Jetzt rufen wir das zweite "W_Complete application" auf
            next_activities = ["W_Complete application"]

        # --- CALLS & VALIDATION ---
        elif completed_activity == "W_Call after offers":
            # 5 Wege XOR
            p_cancel = 0.1
            p_validate = 0.3
            p_complete = 0.3
            p_loop_direct = 0.2
            # Rest ist Ende

            r = random.random()
            if r < p_cancel:
                next_activities = ["A_Cancelled"]
            elif r < p_cancel + p_validate:
                next_activities = ["W_Validate application"]
            elif r < p_cancel + p_validate + p_complete:
                next_activities = ["A_Complete"]
            elif r < p_cancel + p_validate + p_complete + p_loop_direct:
                next_activities = ["W_Call after offers"]
            else:
                # Ende über Call
                self.status = "COMPLETED"
                next_activities = []

        elif completed_activity == "A_Cancelled":
            next_activities = ["O_Cancelled"]

        elif completed_activity == "O_Cancelled":
            next_activities = ["W_Call after offers"]

        elif completed_activity == "A_Complete":
            next_activities = ["W_Call after offers"]

        # --- VALIDATION ---
        elif completed_activity == "W_Validate application":
            # 3 Wege
            p_success = 0.5
            p_return_loop = 0.3
            # Rest: Direkter Loop

            r = random.random()
            if r < p_success:
                next_activities = ["O_Accepted"]
            elif r < p_success + p_return_loop:
                next_activities = ["A_Validating"]
            else:
                next_activities = ["W_Validate application"]

        elif completed_activity == "A_Validating":
            next_activities = ["O_Returned"]

        elif completed_activity == "O_Returned":
            next_activities = ["W_Validate application"]

        elif completed_activity == "O_Accepted":
            next_activities = ["A_Pending"]

        # --- DEINE LOGIK FÜR PENDING ---
        elif completed_activity == "A_Pending":
            # Du sagtest, es geht zurück zur Validation (Loop)
            next_activities = ["W_Validate application"]

            # Hinweis: Achte darauf, dass du hier keine Endlosschleife baust,
            # irgendwann muss "O_Accepted" mal zum Ende führen oder "W_Call after offers"
            # muss den "COMPLETED" Status erreichen.

        # Update Current State
        for act in next_activities:
            self.current_activities.add(act)

        return next_activities

    def _reset_parallel_flags(self):
        self.parallel_path_completed["A_Concept"] = False
        self.parallel_path_completed["W_Complete application"] = False


# InstanceManager Klasse (unverändert)
class InstanceManager:
    def __init__(self):
        self.instances: Dict[str, ProcessInstance] = {}

    def create_instance(self, case_id: str, start_time: float) -> ProcessInstance:
        instance = ProcessInstance(case_id, start_time)
        self.instances[case_id] = instance
        return instance

    def get_instance(self, case_id: str) -> ProcessInstance:
        return self.instances.get(case_id)

    def instance_exists(self, case_id: str) -> bool:
        return case_id in self.instances