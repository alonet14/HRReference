import numpy as np
import utils_v1.butterworth_filter as btwf
import controller.data_ref_controller as data_ref_controller


class RadarDataController:
    def __init__(self, radar_data, fs_radar, ref_data, fs_ref, mtime):
        """
        Handle Data taken from radar
        :param radar_data:
        :param fs_radar: in Hz
        :param ref_data:
        :param fs_ref: in Hz
        :param mtime: in seconds
        """
        self.radar_data = radar_data
        self.ref_data = ref_data
        self.fs_radar = fs_radar
        self.fs_ref = fs_ref
        self.mtime = mtime

    def filted_hr_from_radar_data(self):
        filtered_data = btwf.butter_bandpass_filter(data=self.radar_data, order=5)
        return filtered_data

    def get_window_arr(self):
        return

    def create_window_arr(self):
        ref_data_controller = data_ref_controller.RefDataController(self.ref_data, self.fs_ref, self.mtime)
        window_arr_ref = ref_data_controller.create_window_arr(100, 360)
        window_arr_radar = np.zeros(len(self.radar_data))
        for i, v in enumerate(window_arr_ref):
            index_window_arr_radar = round(i * self.fs_radar / self.fs_ref)
            window_arr_radar[index_window_arr_radar] = v
        return window_arr_radar

    def get_center_window(self, center_from_ref_data=[]):
        rs = map(lambda x: round(x * self.fs_radar / self.fs_ref),
                 center_from_ref_data)
        return rs

    def mark_label(self, window_size=360):
        # wrong=np.zeros(len())
        raw_data = self.radar_data
        peaks_radar = btwf.find_hr(raw_data)
        filtered_data = btwf.butter_bandpass_filter(raw_data)

        window_arr = self.create_window_arr()

        list_upper_window_correct = []
        list_lower_window_correct = []

        list_upper_window_wrong = []
        list_lower_window_wrong = []

        for i, v in enumerate(peaks_radar):
            upper_window = int(v + window_size / 2)
            lower_window = int(v - window_size / 2)
            if lower_window > 0 and upper_window<=len(self.radar_data):
                if window_arr[v] == 1:
                    list_lower_window_correct.append(lower_window)
                    list_upper_window_correct.append(upper_window)
                elif window_arr[v] == 0:
                    list_lower_window_wrong.append(lower_window)
                    list_upper_window_wrong.append(upper_window)

        number_wrong_data = len(list_lower_window_wrong)
        number_correct_data = len(list_lower_window_correct)

        wrong_data = np.empty([number_wrong_data, window_size])
        correct_data = np.empty([number_correct_data, window_size])

        for i in range(0, number_correct_data):
            upper_win_correct = list_upper_window_correct[i]
            lower_win_correct = list_lower_window_correct[i]

            correct_data[i, :] = filtered_data[lower_win_correct:upper_win_correct]

        for i in range(0, number_wrong_data):
            upper_win_wrong = list_upper_window_wrong[i]
            lower_win_wrong = list_lower_window_wrong[i]
            wrong_data[i, :] = filtered_data[lower_win_wrong:upper_win_wrong]

        return {
            'wrong_data': wrong_data,
            'correct_data': correct_data
        }
    def save_window(self, window, window_size=360):
        window_data_matrix = np.zeros((len(window), self.window_size))
        pass
