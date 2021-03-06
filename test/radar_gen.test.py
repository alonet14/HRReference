import plotly.express as px
from controller.radar_gen import RadarGenV2
from utils_v1 import butterworth_filter as btwf
import numpy as np
import pandas as pd
from pandas import DataFrame
import utils_v1.file_utils.file_pandas as fd
# peaks = btwf.find_hr(data=signal_hb_res, fs=radar_gen.fs, order=3, distance=50, prominence=1)
# print(len(peaks)*3)
# time_arr_resp = np.arange(0, len(signal_hb_res), 1)

# fig3 = px.line(x=time_arr_resp, y=radar_gen.location_target, title="test")
# fig3.add_scatter(x=time_arr_resp, y=radar_gen.hb_val, mode='lines')
# fig3.add_scatter(x=time_arr_resp, y=radar_gen.res_val, mode='lines')
# fig3.show()
#
# fig = px.line(x=time_arr_resp, y=filtered_signal, title="test")
# fig.add_scatter(x=peaks, y=filtered_signal[peaks], mode='markers')
# fig.show()
#
# fig2 = px.line(x=time_arr_resp, y=signal_hb_res, title="test")
# fig2.add_scatter(x=time_arr_resp, y=hb_signal, mode='lines')
# fig2.show()

import utils_v1.file_utils.file_pandas as fp

file_path = '../data/recorded_data_gen/data.csv'
# fp.write_csv_file(file_path, ['value', 'label'])

fre_hb_gen = 50

#chay tu 50 - 140
for i in range(90):
    # label=np.zeros(90)
    # label[i]=1
    radar_gen = RadarGenV2(fs=100, mtime=20)
    data_list=[]
    quantity_signal_per_label=50
    for j in range(quantity_signal_per_label):
        signal_hb_res = radar_gen.i_signal(fhb=fre_hb_gen, te=2, ti=2, snr=0)
        filtered_signal = btwf.butter_bandpass_filter(data=signal_hb_res, fs=radar_gen.fs, order=3)
        filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
        labeled_filtered_signal=filtered_signal
        data_list.append(labeled_filtered_signal)
    data_frame = DataFrame(data_list)
    path_laptop = 'D:\\lab_project\\hr_processing\\data\\recorded_data_gen\\data{0}.csv'.format("_"+str(fre_hb_gen))
    path_lab = ''
    fd.create_file(path_laptop)
    data_frame.to_csv(path_laptop, mode='w', header=True)
    fre_hb_gen += 1



