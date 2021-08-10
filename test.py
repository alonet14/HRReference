import os

import numpy as np
from utils_v1.config_path import project as pr
from utils_v1.pandas_handler import read_file_pandas
import controller.data_ref_controller as data_ref_controller
import controller.data_radar_controller as data_radar_controller
import plotly.express as px
import pandas as pd
import utils_v1.butterworth_filter as btwf
import controller.signal_generate as sng

# ========config=============
root_pro = pr.get_root_file(__file__)
data_folder = str(os.path.join(root_pro, "data/reference"))
window_size = 360
fs = 250
mid_area_size = 40


def get_data(data_folder_path, file_number=0):
    list_data_file = read_file_pandas.get_list_file(data_folder_path)
    data = list(read_file_pandas.read_file(list_data_file[file_number], 1).index)
    data_np = np.asarray(data, dtype=np.float64)
    return data_np


def plot(x, y, peaks, window, name):
    fig = px.line(x=x, y=y, title=name)
    fig.add_scatter(x=peaks, y=y[peaks], mode='markers')
    fig.add_scatter(x=x, y=window)
    fig.show()


def plot_with_peaks(x, y, peaks, name):
    fig = px.line(x=x, y=y, title=name)
    fig.add_scatter(x=peaks, y=y[peaks], mode='markers')
    fig.show()


def plot2(x, y, y2, name):
    fig = px.line(x=x, y=y, title=name)
    fig.add_scatter(x=x, y=y2)
    fig.show()


def ecg_peaks(data=[], fs=250):
    from lib.ecg_processing.ecgdetectors import Detectors
    detectors = Detectors(fs)
    r_peaks = detectors.engzee_detector(data)
    return r_peaks


def test():
    #
    data = get_data(data_folder, 0)
    data = data[2000:6999]
    ref_data_controller = data_ref_controller.RefDataController(
        ref_data=data,
        fs=250,
        mtime=10)
    window_arr = ref_data_controller.create_window_arr()

    mtime = np.arange(0, len(window_arr), 1)
    peaks = ecg_peaks(data)
    data = data / np.max(data)
    plot(x=mtime, y=data, peaks=peaks, window=window_arr, name='tín hiệu tham chiếu')

    # radar_data

    file_path = '/home/nvh/lab_project/hr_processing/data/radar/data_test_1626256597.csv'
    radar_data = pd.read_csv(file_path)
    radar_data = np.asarray(list(radar_data.values))
    radar_data = radar_data[:, 0]
    radar_data = radar_data[4450:6450]
    radar_data_controller = data_radar_controller.RadarDataController(
        radar_data=radar_data,
        fs_radar=100,
        ref_data=data,
        fs_ref=250,
        mtime=10
    )
    window_radar = radar_data_controller.create_window_arr()
    peaks_radar = btwf.find_hr(radar_data)

    mtime = np.arange(0, len(radar_data), 1)

    filtered_data = btwf.butter_bandpass_filter(data=radar_data)

    filtered_data = filtered_data / np.max(filtered_data)
    label = radar_data_controller.mark_label(window_size=360)
    wrong_label = label['wrong_data']
    correct_label = label['correct_data']
    plot(x=mtime, y=filtered_data, window=window_radar, peaks=peaks_radar, name='tín hiệu radar')

    fig = px.line(x=[x for x in range(0, 360)], y=wrong_label[0, :], title='label=0')
    fig2 = px.line(x=[x for x in range(0, 360)], y=correct_label[0, :], title='label=1')

    fig.show()
    fig2.show()


if __name__ == '__main__':
    band_hr = (0.83, 2.33)
    band_rr = (0.167, 3.167)
    fs = 100
    mtime = 15
    snr = -10
    sig_gen = sng.SignalGenerate(band_hr=band_hr, band_rr=band_rr, snr=snr, fs=fs, mtime=mtime)
    data = sig_gen.gen_raw_singal()

    mtime_x = np.arange(0, mtime * 100, 1)
    filter_data = btwf.butter_bandpass_filter(data[:, 6])
    peaks = btwf.find_hr(data[:, 6])

    hb = sig_gen.hb

    hb_noise = sig_gen.hb_noise
    plot2(mtime_x, hb[:, 6], hb_noise[:,6], 'test')
    plot_with_peaks(x=mtime_x, y=filter_data/np.max( np.abs(filter_data)), peaks=peaks, name='radar data with peak')

    sig_handle=sng.SignalHandle()
    for i in range(0, 5):
        sig_handle.save_data_with_label(filter_data, 50)
