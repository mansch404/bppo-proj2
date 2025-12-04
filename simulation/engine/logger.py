import csv
import shutil
import pm4py
import pandas as pd

class Logger(object):

    ATTRIBUTES =  ['Action', 'org:resource', 'concept:name',
                   'EventOrigin', 'EventID', 'lifecycle:transition',
                   'time:timestamp', 'case:LoanGoal',
                   'case:ApplicationType', 'case:concept:name',
                   'case:RequestedAmount', 'FirstWithdrawalAmount',
                   'NumberOfTerms', 'Accepted', 'MonthlyCost',
                   'Selected', 'CreditScore', 'OfferedAmount', 'OfferID'
                    ]

    def __init__(self, filename: str):
        self.filename = filename
        self.filepath = "/data/sim_logs/" + filename

        # Initializing the log and writing the attribute columns
        with open(self.filepath, 'w', newline='', encoding='utf-8') as log:
            writer = csv.DictWriter(log, fieldnames=self.ATTRIBUTES)
            writer.writeheader()

    def log_event(self, action, concept_name, org_source, eventOrigin, eventID, lifecyleTransition,
             timestamp, loanGoal, applicationType, caseID, requestedAmount,
             firstWithdrawal, numberOfTermns, accepted, monthlyCost, selected,
             creditScore, offeredAmount, offerID):


        event_data = {
            "Action": action,
            "org:resource": org_source,
            "concept:name": concept_name,
            "EventOrigin": eventOrigin,
            "EventID": eventID,
            "lifecycle:transition": lifecyleTransition,
            "time:timestamp": timestamp,
            "case:LoanGoal": loanGoal,
            "case:ApplicationType": applicationType,
            "case:concept:name": caseID,
            "case:RequestedAmount": requestedAmount,
            "FirstWithdrawalAmount": firstWithdrawal,
            "NumberOfTerms": numberOfTermns,
            "Accepted": accepted,
            "MonthlyCost": monthlyCost,
            "Selected": selected,
            "CreditScore": creditScore,
            "OfferedAmount": offeredAmount,
            "OfferID": offerID
        }

        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.ATTRIBUTES)
            writer.writerow(event_data)



    def export_log_to_cvs(self, output_path: str = None):

        if output_path is None:
            output_path = self.filename

            try:
                shutil.copy(self.filepath, output_path)
                print("Successfully exported log file to csv.")
            except FileNotFoundError:
                print("Log file not found.")

    def export_log_to_xes(self,output_path: str = None ):

        if output_path is None:
            output_path = self.filename.replace('.csv', '.xes')
            if not output_path.endswith('.xes'):
                output_path += '.xes'

        # Read csv file and convert it to dataframe
        df = pd.read_csv(self.filepath)

        # Check correct timestamp format
        if 'time:timestamp' in df.columns:
            try:
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
            except Exception as e:
                print(f"Error: {e}")

        try:
            # Convert dataframe into .xes file
            pm4py.write_xes(df, output_path, case_id_key='case:concept:name')
            print("Successfully exported log to xes.")
        except Exception as e:
            print("Failed to export XES:", e)





