import pathlib

from scipy.fft import fft, fftfreq, ifft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import controller.signal_generate as sng
import plotly.express as px
import os
import pathlib as pl

def read_txt_data(file_path='/home/nvh/lab_project/hr_processing/data/radar/data_test_1626256898.csv'):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f).values
        df = df[:, 0]
        f.close()
        return df


def fourier_transform(all, start=0, stop=10):
    data = read_txt_data()
    if not all:
        data = data[start:stop]
    data_f = fft(data)
    return data_f



def test_signal_render():
    #============noise in hr band========
    am_noise_in_hr_band = 0.006
    ran_hr_freq = 0.83 + np.random.rand() * (2.33 - 0.83)
    ran_rr_freq = 0.167 + np.random.rand() * (0.367 - 0.167)

    radar_gen = sng.RadarGen(ran_hr_freq, ran_rr_freq)
    am_arr, freq_arr = radar_gen.gen_noise_in_hr_band(am_noise_in_hr_band)
    # fig = px.scatter(x=np.arange(0, len(freq_arr), 1), y=freq_arr)
    # fig.show()

    #=========gen movement signal========
    i_signal=radar_gen.radar_data_gen()
    fs = 100
    mtime=np.arange(0, len(i_signal)/fs, 1/fs)

    i_signal_fig=px.line(x=mtime, y=i_signal, title='i signal')
    i_signal_fig.add_scatter(x=mtime, y=radar_gen.lung_signal)
    i_signal_fig.add_scatter(x=mtime, y=radar_gen.heart_signal)
    # i_signal_fig.add_scatter(x=mtime, y=radar_gen.noise[0])
    print(radar_gen.noise)
    i_signal_fig.show()

    #fourier transform
    data_f = fft(i_signal)
    xf = fftfreq(len(data_f), 1 / fs)
    size_data=len(data_f)
    freq_spec = px.line(x=xf, y=2.0/size_data*np.abs(data_f), title='frequency domain')

    freq_spec.show()





def analyse_real_data(index):
    parent_folder=pathlib.Path(__file__).parent
    parent_folder = str(parent_folder)+"/data/radar"
    list_path=[]
    for root, _, filenames in os.walk(parent_folder):
        for f in filenames:
            file_path=root + '/' + f
            list_path.append(file_path)
    print(list_path)
    i_path=index
    get_file = list_path[i_path]

    #==========================analyse real data =====================
    data = read_txt_data(file_path=get_file)
    print(data)

    fs=100
    time=np.arange(0, len(data)/fs, 1/fs)
    fig = px.line(x=time, y=data, title='radar data')
    fig.show()

    #==================analyse frequency spectrum============
    data_f=fft(data)
    len_data=len(data_f)
    xf=fftfreq(len(data), 1/fs)
    fig2=px.line(x=xf, y=2/len_data*np.abs(data_f), title='frequency domain')
    fig2.show()



if __name__ == '__main__':
    index_file=1

    analyse_real_data(index_file)

    # test_signal_render()
