import numpy as np
import matplotlib.pyplot as plt
import pathlib
from ecgdetectors import Detectors
import plotly.express as px

from utils_v1.pandas_handler import read_file_pandas


def plot(x, y):
    fig = px.line(x=x, y=y,title='ECG signal')

    fig.show()

if __name__ == '__main__':
    current_dir = pathlib.Path(__file__).resolve()

    example_dir = 'E:\lab_project\hr_processing\data'

    list_data_file = read_file_pandas.get_list_file(example_dir)
    data = list(read_file_pandas.read_file(list_data_file[2], 1).index)
    data_np = np.asarray(data, dtype=np.float64)
    data_np = data_np[1000:3000]

    # unfiltered_ecg_dat = np.loadtxt(example_dir)
    # unfiltered_ecg = unfiltered_ecg_dat[:, 0]
    unfiltered_ecg = data_np
    fs = 360

    detectors = Detectors(fs)

    #r_peaks = detectors.two_average_detector(unfiltered_ecg)
    #r_peaks = detectors.matched_filter_detector(unfiltered_ecg,"templates/template_250hz.csv")
    #r_peaks = detectors.swt_detector(unfiltered_ecg)
    r_peaks = detectors.engzee_detector(unfiltered_ecg)
    #r_peaks = detectors.christov_detector(unfiltered_ecg)
    #r_peaks = detectors.hamilton_detector(unfiltered_ecg)
    #r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)


    plt.figure()
    time = np.arange(0, len(unfiltered_ecg),  1)

    plt.plot(time, unfiltered_ecg)
    plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'x')

    # plt.title('Detected R-peaks')
    plt.show()
