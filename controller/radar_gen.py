import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
import control
from komm import AWGNChannel

from utils_v1 import butterworth_filter as btwf


class RadarGenV2():
    def __init__(self, fs=100, mtime=20):
        self.fs = fs
        self.mtime = mtime

    def gen_respirator_v1(self, **kargs):
        """
            refernece: "High Accuracy Heartbeat Detection from
            CW-Doppler Radar Using Singular Value Decomposition and Matched Filter"
            - authors: Yuki Iwata, Han Trong Thanh, Guanghao Sun and Koichiro Ishibashi
        """
        # Kb = 10 * pow(10, -3)
        # Ti = 2 * pow(10, -3)
        # Te = 3 * pow(10, -3)
        # T = Ti + Te
        # ===config parameters====
        Kb = kargs['kb']
        Ti = kargs['ti']
        Te = kargs['te']
        T = Ti + Te
        mtime = self.mtime
        t_c = kargs['tc']

        time_arr = np.arange(0, mtime, 0.0001)
        data_resp = []
        res_function_model = None

        for t in time_arr:
            # time in period
            tip = t - T * int(t / T)
            if tip >= 0 and tip <= Ti:
                res_function_model = (-Kb / (Ti * Te)) * tip * tip + (Kb * T / (Ti * Te)) * tip
                # res_function_model = 0
            elif tip > Ti and tip <= T:
                res_function_model = (Kb / (1 - np.exp(-Te / t_c))) * tip * tip * (
                        np.exp(-(tip - Te) / t_c) - np.exp(-Te / t_c))

                # res_function_model = 0
            data_resp.append(res_function_model)
        return data_resp

    def gen_respirator_v2(self, **kargs):
        """
        reference: System Modeling and Signal Processing of Microwave Doppler Radar for Cardiopulmonary Sensing
        """
        am = kargs['am']
        freq = kargs['freq']
        phase = kargs['phase']
        mtime = kargs['mtime']

        time_arr = np.arange(0, mtime, 0.01)
        data_resp = []
        for t in time_arr:
            res_val = am * np.sin(2 * np.pi * freq * t + phase)
            data_resp.append(res_val)
        return data_resp

    def gen_heartbeat_movement_signal_v1(self, **kargs):
        am = kargs['am']
        freq = kargs['freq']
        phase = kargs['phase']
        mtime = kargs['mtime']

        time_arr = np.arange(0, mtime, 0.01)
        data_heart = []
        for t in time_arr:
            heart_val = am * np.sin(2 * np.pi * freq * t + phase)
            data_heart.append(heart_val)
        return data_heart

    def location_xt(self):
        mtime = 20
        am_res = 10 * pow(10, -3)
        freq_res = 0.25
        phase_res = 2 * np.pi * np.random.rand()

        am_hb = 0.8 * 10 * pow(10, -3)
        freq_hb = 1.2
        phase_hb = 2 * np.pi * np.random.rand()

        res_val = self.gen_respirator_v2(am=am_res, freq=freq_res, phase=phase_res, mtime=mtime)
        hb_val = self.gen_heartbeat_movement_signal_v1(am=am_hb, freq=freq_hb, phase=phase_hb, mtime=mtime)

        signal_hb_res = am_res + am_hb - (np.asarray(res_val) + np.asarray(hb_val))
        return signal_hb_res

    def i_signal(self):
        location_target = self.location_xt()
        c = 3 * pow(10, 8)
        f_carrying_wave = 24 * pow(10, 9)
        wavelength = c / f_carrying_wave
        print(location_target[1] / wavelength)
        constant_phase_shift = 4 * np.pi * 0.3 / wavelength
        i_data = []
        snr = -30
        linear_snr = control.db2mag(snr)
        awgn = AWGNChannel(snr=linear_snr, signal_power='measured')
        for instantous_location in location_target:
            i_val = 5.5 * np.cos(constant_phase_shift + 4 * np.pi * instantous_location / wavelength)
            i_data.append(i_val)
        i_data = awgn(i_data)
        return i_data

    def filter_i_signal(self):
        i_signal = self.i_signal()
        filtered_signal = btwf.butter_bandpass_filter(data=i_signal)
        return filtered_signal


radar_gen = RadarGenV2()
signal_hb_res = radar_gen.i_signal()
filtered_signal = btwf.butter_bandpass_filter(data=signal_hb_res, order=5)
filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
peaks = btwf.find_hr(data=signal_hb_res)

time_arr_resp = np.arange(0, len(signal_hb_res), 1)
fig = px.line(x=time_arr_resp, y=signal_hb_res, title="test")
# fig.add_scatter(x=peaks, y=filtered_signal[peaks], mode='markers')
fig.show()

# fig=px.scatter(x=time_arr_resp, y=data_res)
# fig.show()

# ======hb=====
# data_hb=[]
# disp_hb=0.5*pow(10, -3)
# fhr=120
# for t in time_arr:
#     hb_function_model=disp_hb*np.sin(2*np.pi*fhr*t)
#     data_hb.append(hb_function_model)
#
# fig = plt.plot(time_arr, data_res)
# plt.show()
