import numpy as np
import lib.ecg_processing.ecgdetectors as ecgdetector


class RefDataController:
    def __init__(self, ref_data, fs, mtime):
        self.ref_data = ref_data
        self.fs = fs
        self.mtime = mtime

    def __find_r_peaks(self):
        detector = ecgdetector.Detectors(self.fs)
        r_peaks = detector.engzee_detector(self.ref_data)
        return r_peaks

    def create_window_arr(self, mid_area_size=100, window_size=360):
        window_arr = np.zeros(len(self.ref_data))
        r_peaks = self.__find_r_peaks()
        for index, value in enumerate(r_peaks):
            lower = int(value - mid_area_size / 2) + 1
            upper = int(value + mid_area_size / 2)
            window_arr[lower:upper] = 1
        return window_arr


