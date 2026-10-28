from data import get_data
from analysis import analysis_pipeline

def run_analysis():
    get_data()
    analysis_pipeline()

run_analysis()