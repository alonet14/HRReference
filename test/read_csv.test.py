import csv
import pandas as pd
with open("../data/recorded_data_gen/data.csv", 'r') as f:
    data=pd.read_csv(f)
    print(data)
