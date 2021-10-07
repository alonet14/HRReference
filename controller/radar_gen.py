import numpy as np
import control
from komm import AWGNChannel
from utils_v1 import butterworth_filter as btwf


class RadarGenV2():
    def __init__(self, fs=100, mtime=20):
        """
        fs: sample rate
        mtime: measured time
        """

        self.fs = fs
        self.mtime = mtime

    def gen_respirator_v1(self, **kargs):
        """
            refernece: "High Accuracy Heartbeat Detection from
            CW-Doppler Radar Using Singular Value Decomposition and Matched Filter"
            - authors: Yuki Iwata, Han Trong Thanh, Guanghao Sun and Koichiro Ishibashi
        """
        # ===config parameters====
        Kb = kargs['kb']
        Ti = kargs['ti']
        Te = kargs['te']
        T = Ti + Te
        mtime = self.mtime
        t_c = kargs['tc']

        time_arr = np.arange(0, mtime, 1 / self.fs)
        data_resp = []
        res_function_model = None

        for t in time_arr:
            # time in period
            tip = t - T * int(t / T)
            if tip >= 0 and tip <= Ti:
                res_function_model = (-Kb / (Ti * Te)) * tip * tip + (Kb * T / (Ti * Te)) * tip
                # res_function_model = 0
            elif tip > Ti and tip <= T:
                res_function_model = (Kb / (1 - np.exp(-Te / t_c))) * (
                        np.exp(-(tip - Te) / t_c) - np.exp(-Te / t_c))
            data_resp.append(res_function_model)
        return data_resp

    def gen_respirator_v2(self, **kargs):
        """
        reference: System Modeling and Signal Processing of Microwave Doppler Radar for Cardiopulmonary Sensing
        """
        am = kargs['am']
        freq = kargs['freq']
        phase = kargs['phase']
        mtime = self.mtime

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
        mtime = self.mtime

        time_arr = np.arange(0, mtime, 0.01)
        data_heart = []
        for t in time_arr:
            heart_val = am * np.sin(2 * np.pi * freq * t + phase)
            data_heart.append(heart_val)
        return data_heart

    def vibration_target(self, am_hb=0.07 * 10 * pow(10, -3), fhb=1.0, am_res=10 * pow(10, -3), te=2, ti=2):
        phase_hb = 2 * np.pi * np.random.rand()
        res_val = self.gen_respirator_v1(am=am_res, kb=am_res, ti=ti, te=te, tc=0.8)
        hb_val = self.gen_heartbeat_movement_signal_v1(am=am_hb, freq=fhb, phase=phase_hb)
        signal_hb_res = am_res + am_hb - (np.asarray(res_val) + np.asarray(hb_val))
        return signal_hb_res

    def i_signal(self, normal_distance_target=30 * pow(10, -2), snr=-10):
        # unit: meter
        # snr in db
        location_target = self.vibration_target()
        self.location_target = location_target

        c = 3 * pow(10, 8)
        f_carrying_wave = 24 * pow(10, 9)
        wavelength = c / f_carrying_wave
        constant_phase_shift = 4 * np.pi * normal_distance_target / wavelength
        i_data = []

        linear_snr = control.db2mag(snr)
        awgn = AWGNChannel(snr=linear_snr, signal_power='measured')
        for instantous_location in location_target:
            i_val = 3.5 * np.cos(constant_phase_shift + 4 * np.pi * instantous_location / wavelength)
            i_data.append(i_val)
        i_data = awgn(i_data)
        return i_data

    def filter_i_signal(self):
        normal_distance_target = 30 * pow(10, -2)
        snr = -10
        i_signal = self.i_signal(normal_distance_target, snr)
        filtered_signal = btwf.butter_bandpass_filter(data=i_signal, order=3)
        return filtered_signal
