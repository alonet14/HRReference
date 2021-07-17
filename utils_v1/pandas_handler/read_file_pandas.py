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


