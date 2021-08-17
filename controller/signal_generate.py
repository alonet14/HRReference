import errno
import pathlib
import random

import pandas as pd
import numpy as np
from komm import AWGNChannel
import control
import os
import pathlib as pl
from csv import writer
import random as rd


class SignalGenerate():
    def __init__(self, band_hr, band_rr, snr, fs, mtime):
        self.band_hr = band_hr
        self.band_rr = band_rr
        self.fs = fs
        self.mtime = mtime
        self.snr = snr
        self.hb = None
        self.hb_noise = None

    def awgn(self, s, SNRdB, L=1):
        # author - Mathuranathan Viswanathan (gaussianwaves.com
        # This code is part of the book Digital Modulations using Python
        from numpy import sum, isrealobj, sqrt
        from numpy.random import standard_normal
        """
        AWGN channel
        Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
        returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
        Parameters:
            s : input/transmitted signal vector
            SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
            L : oversampling factor (applicable for waveform simulation) default L = 1.
        Returns:
            r : received signal vector (r=s+n)
        """
        gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
        if s.ndim == 1:  # if s is single dimensional vector
            P = L * sum(abs(s) ** 2) / len(s)  # Actual power in the vector
        else:  # multi-dimensional signals like MFSK
            P = L * sum(sum(abs(s) ** 2)) / len(s)  # if s is a matrix [MxN]
        print(len(s))
        N0 = P / gamma  # Find the noise spectral density
        if isrealobj(s):  # check if input is real/complex object type
            n = sqrt(N0 / 2) * standard_normal(s.shape)  # computed noise
        else:
            n = sqrt(N0 / 2) * (standard_normal(s.shape) + 1j * standard_normal(s.shape))
        r = s + n  # received signal

        return r

    def _signal_hr_without_noise(self):
        numsig = self.fs * self.mtime
        center_freq = (np.max(self.band_hr) - np.min(self.band_hr)) * np.random.random_sample((numsig, 1)) + min(
            self.band_hr)
        sample = np.arange(0, numsig, 1)
        t = sample / self.fs
        print(np.pi)
        # hb = cos(
        # 2 * pi * (centerfreq * t - 0.1 * cos(2 * pi * t / 100 + 2 * pi * rand) / (2 * pi / 100)) + 2 * pi * rand);
        hb = np.cos(2 * np.pi *
                    (center_freq * t - 0.1 * np.cos(2 * np.pi * t / 100)) /
                    (2 * np.pi / 100)
                    ) + 2 * np.pi * np.random.random_sample((numsig, numsig)
                                                            )
        self.hb = hb
        return hb

    def gen_hr_with_noise(self):
        s = self._signal_hr_without_noise()
        # r = self.awgn(s=s, SNRdB=self.snr, L=0.1)
        linear_snr = control.db2mag(self.snr)
        awgn = AWGNChannel(snr=linear_snr, signal_power='measured')
        column_size = self.mtime * self.fs
        noise_matrix = np.empty_like(s)
        for i in range(0, column_size):
            data = s[i, :]
            r = awgn(data)
            noise_matrix[i, :] = r
        self.hb_noise = s + noise_matrix

        return s

    def _signal_rr(self):
        numsig = self.fs * self.mtime
        frr = (np.max(self.band_rr) - np.min(self.band_rr)) * np.random.random_sample((numsig, 1)) + min(self.band_rr)
        sample = np.arange(0, numsig, 1)
        t = sample / self.fs
        rb = 10 * np.cos(2 * np.pi * frr * t)
        return rb

    def gen_raw_singal(self):
        # r = self.gen_hr_with_noise() + self._signal_rr()
        r = self._signal_hr_without_noise()
        return r


class RadarGen():
    def __init__(self, hr_freq, rr_freq):
        self.hr_freq = hr_freq
        self.rr_freq = rr_freq

    def radar_data_gen(self):
        AI = 2
        wavelength = 0.0125
        freq_heart = 0.83 + np.random.rand() * (2.33 - 0.83)
        freq_lung = 0.167 + np.random.rand() * (0.367 - 0.167)
        heart_am = 0.1
        lung_am = 3
        t = np.arange(0, 20, 0.01)
        ran_phase_lung = 2 * np.pi * np.random.rand()
        ran_phase_heart = 2 * np.pi * np.random.rand()
        noise = np.empty((2, len(t)))
        for i in range(0, len(t)):
            ran_frequency = 12 * np.random.rand()
            noise[0, i] = ran_frequency
            ran_am = 5 * np.random.rand()
            noise[1, i] = ran_am

        i_signal = lung_am * np.sin(2 * np.pi * freq_lung * t + ran_phase_lung) + heart_am * np.sin(
            2 * np.pi * freq_heart * t + ran_phase_heart)

        self.lung_signal=lung_am * np.sin(2 * np.pi * freq_lung * t + ran_phase_lung)
        self.heart_signal=heart_am * np.sin(2 * np.pi * freq_heart * t + ran_phase_heart)
        print(np.sin(np.pi/2))
        # self.noise=np.empty_like((1, len(t)))
        noise_total=np.empty((1, len(t)), dtype='float64')

        for j in range(0, len(t)):
            noise_total+=noise[1, j] * np.sin(2 * np.pi * noise[0, j]*t + 2 * np.pi * np.random.rand())
        self.noise=noise_total
        noise_total=noise_total/np.max(np.abs(noise_total))*7
        i_signal+=noise_total[0]
        print(self.noise)
        return i_signal

    def gen_hr_freq(self, center_fre):
        pass

    def gen_noise_in_hr_band(self, am_noise=0.006):
        mtime = 1689
        numsig = 1689
        am_arr = np.empty(numsig)
        freq_arr = np.empty(numsig)

        for i in range(0, mtime):
            amptitude = np.random.rand() * am_noise
            freq = np.random.rand() * (3 - 0.83) + 0.83
            am_arr[i] = amptitude
            freq_arr[i] = freq

        return am_arr, freq_arr

    def gen_freq_spectrum(self):
        pass

    def gen_freq_noise_out_hr_band(self):
        pass


class SignalHandle():
    def search_label_file(self, file_name=""):
        """
        if label file exist:
            return path of the file
        else:
            create file and return path of the file
        :param file_name: name of the file need to find
        :return: path of the file
        """
        from sys import platform
        root_file = str(pl.Path(__file__).parent.parent)
        train_file_path = ""
        if platform == "linux" or platform == "linux2":
            train_file_path = "/train_data/{}/{}".format(file_name, file_name + '.csv')
        elif platform == "win32":
            train_file_path = "\\train_data\\{}\\{}".format(file_name, file_name + '.csv')
        train_file_path = root_file + train_file_path

        if not (os.path.isdir(train_file_path) or os.path.isfile(train_file_path)):
            try:
                os.makedirs(os.path.dirname(train_file_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            with open(train_file_path, 'w') as f:
                f.close()

        return train_file_path

    def save_data_with_label(self, data, label):
        """
        :param data:
        :param label: int, heart rate number
        :return: void
        """
        file_name = 'label_{}'.format(label)
        file_path = self.search_label_file(file_name)
        file_path = pathlib.Path(file_path)
        with open(file_path, 'a', newline='') as f:
            write_csv = writer(f)
            write_csv.writerow(data)
            f.close()
