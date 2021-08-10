import os
# from src.utils_v1.pandas_handler import read_file_pandas
from utils_v1.config_path import project as pr
import plotly.express as px
import numpy as np
import pywt
import scipy.signal as sig

from utils_v1.pandas_handler import read_file_pandas

root_pro = pr.get_root_file(__file__)
data_folder = str(os.path.join(root_pro, "data/reference"))

def plot(x, y):
    fig = px.line(x=x, y=y, title='custoam tick label')
    fig.show()

def butter_lowpass(cutoff, fs, order=5)    :
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y =sig.lfilter(b, a, data)
    return y

if __name__ == '__main__':
    list_data_file = read_file_pandas.get_list_file(data_folder)
    data = list(read_file_pandas.read_file(list_data_file[2], 1).index)
    data_np = np.asarray(data, dtype=np.float64)
    data_np = data_np[1000:3000]
    measure_time = np.arange(0, len(data_np), 1)

    a = pywt.families(True)
    coeffs = pywt.wavedec(data_np, 'bior3.7', level=3)
    b = pywt.waverec(coeffs, 'bior3.7')
    fs = 360
    fc = 0.667
    data_filtered = butter_lowpass_filter(data=data_np, cutoff=fc, fs=fs, order=3)
    plot(measure_time, data_filtered)
