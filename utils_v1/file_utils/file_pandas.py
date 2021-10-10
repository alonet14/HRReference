import csv
def get_list_file(folder_path=""):
    import os
    rs = []
    for root, _dir, _filenames in os.walk(folder_path):
        for filename in _filenames:
            rs.append(os.path.join(folder_path, filename))
    return rs

def read_file(file_path, column):
    import pandas as pd
    return pd.read_excel(file_path, index_col=column)

def read_csv_file(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def write_csv_file(file_path, row):
    with open(file_path, 'w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(row)
    f.close()


def create_file(file_path):
    import os
    if not os.path.isfile(file_path):
        f=open(file_path, mode='x')
        f.close()
