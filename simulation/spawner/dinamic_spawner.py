import pandas as pd
import fitter
import pickle
import scipy.stats as stats

class DinamicSpawner():

    def __init__(self):
        self.model = None