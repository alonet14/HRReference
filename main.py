import pandas as pd
import pathlib as pl
import os
from utils_v1.pandas_handler import read_file_pandas
from utils_v1.config_path import project as pr
import plotly.express as px
import numpy as np

root_pro = pr.get_root_file(__file__)
data_folder = str(os.path.join(root_pro, "data"))

if __name__ == '__main__':

    list_data_file = read_file_pandas.get_list_file(data_folder)
    data = read_file_pandas.read_file(list_data_file[0], 1).index
    data = list(data)
    data_np = np.asarray(data, dtype=np.float64)
    data_np = data_np[:10000]
    measure_time = np.arange(0, len(data_np), 1)
    df = px.data.stocks()
    fig = px.line(df, x=measure_time, y=data_np,
                  title='custom tick label')
    # fig.update_xaxes(
    #     dtick="M1",
    #     tickformat="%b\n%Y"
    # )
    fig.show()


