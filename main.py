import os
# from src.utils_v1.pandas_handler import read_file_pandas
from utils_v1.config_path import project as pr
import plotly.express as px
import numpy as np
import pywt
import scipy.signal as sig
from utils_v1.pandas_handler import read_file_pandas
from utils_v1 import butterworth_filter as btwf
root_pro = pr.get_root_file(__file__)
data_folder = str(os.path.join(root_pro, "data/reference"))


def plot(x, y, peaks):
    fig = px.line(x=x, y=y, title='custoam tick label')
    fig.add_scatter(x=peaks, y=y[peaks], mode='markers')
    fig.show()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a



def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


def ecg_peaks(data=[], fs=250):
    from lib.ecg_processing.ecgdetectors import Detectors
    detectors = Detectors(fs)
    r_peaks = detectors.engzee_detector(data)
    return r_peaks


def labeling(ecg_data=[],data_from_radar=[], fs=250, mid_area=40, window_size=360):
    """
    :param ecg_data:
    :param type: 0: P peak
    1: Q peak
    2: R peak
    :return: none
    """
    from lib.ecg_processing.ecgdetectors import Detectors
    T = 100
    label_arr = np.zeros([1, len(ecg_data)])
    # init label
    cw1_wrong = np.zeros([len(ecg_data) * 30, 1])
    cw2_wrong = np.zeros([len(ecg_data) * 30, 1])
    cw1_correct = np.zeros([len(ecg_data) * 30, 1])
    cw2_correct = np.zeros([len(ecg_data) * 30, 1])
    hb_filt = np.zeros((len(ecg_data), T * fs))
    k1 = 1
    k2 = 1
    k_dif = np.zeros([len(ecg_data), 1])

    # detect peaks
    for i in range(1, len(ecg_data)):
        detectors = Detectors(fs)
        r_peaks = detectors.engzee_detector(data_np)
        differentiate_array=np.zeros(1, T*fs)
        hb_filt[i,:]=btwf.butter_bandpass_filter(data_from_radar, )
        for index, v in enumerate(r_peaks):
            if v - mid_area / 2 < 1:
                d1 = 1
                d2 = v + mid_area / 2 - 1
            elif v + mid_area / 2 - 1 > T * fs:
                d2 = T * fs
                d1 = v - mid_area / 2
            else:
                d1 = v - mid_area / 2
                d2 = v - mid_area / 2 - 1
            label_arr[d1:d2] = 1
        center_window = r_peaks
        done1 = 1
        done0 = 0
        for index, v in center_window:
            if done1 + done0 < 2:
                cw = v
                cw1 = v - window_size / 2
                cw2 = v + window_size / 2 - 1
                if cw > window_size / 2 + 1 and cw < T * fs - window_size / 2 + 2:
                    if label_arr(v) == 1 and done1 == 0:
                        cw1_correct[k1] = cw1
                        cw2_correct[k2] = cw2
                        k1 += 1
                        if done0 == 1:
                            k_dif[index] = 2
                        else:
                            k_dif[index] = 1
                        done1 = 1
                    elif label_arr[cw] == 0 and done0 == 0:
                        cw1_wrong[k2] == cw1
                        cw2_wrong[k2] == cw2
                        k2 += 1
                        if done1 == 1:
                            k_dif[index] = 2
                        else:
                            k_dif[index] = 0
                        done0 = 1
            else:
                break

    cw1_wrong = cw1_wrong[cw1_wrong[:, 2], :]
    cw1_correct = cw1_correct[cw1_correct[:, 2], :]
    cw1_correct[(len(cw1_wrong) + 1):len(cw1_correct), :] = []
    cw1_wrong[(len(cw1_correct) + 1):len(cw1_wrong), :] = []
    cw2_wrong = cw2_wrong[cw2_wrong[:, 2], :]
    cw2_correct = cw2_correct[cw2_correct[:, 2], :]
    cw2_correct[(len(cw2_wrong) + 1):len(cw2_correct), :] = []
    cw2_wrong[(len(cw2_correct)) + 1:len(cw2_wrong), :] = []

    wrong = np.zeros((len(cw1_wrong), window_size))
    correct = np.zeros((len(cw1_wrong), window_size))
    i1 = 1
    i2 = 1
    for index in range(1, len(ecg_data)):
        if k_dif[index] == 1 and i1 <= len(cw1_correct):
            correct[i1, :] =


if __name__ == '__main__':
    list_data_file = read_file_pandas.get_list_file(data_folder)
    data = list(read_file_pandas.read_file(list_data_file[0], 1).index)
    data_np = np.asarray(data, dtype=np.float64)
    data_np = data_np[1000:3000]
    measure_time = np.arange(0, len(data_np), 1)

    a = pywt.families(True)
    coeffs = pywt.wavedec(data_np, 'bior3.7', level=3)
    b = pywt.waverec(coeffs, 'bior3.7')
    fs = 360
    fc = 0.667
    data_filtered = butter_lowpass_filter(data=data_np, cutoff=fc, fs=fs, order=3)

    r_peaks = ecg_peaks(data_np)

    plot(measure_time, data_np, r_peaks)
    # plt.plot(measure_time, data_np, 'r.')
    # plt.show()

    # x and y given as array_like objects
    import plotly.express as px

    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.show()
