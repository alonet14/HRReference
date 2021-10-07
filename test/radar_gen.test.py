import plotly.express as px
from controller.radar_gen import RadarGenV2
from utils_v1 import butterworth_filter as btwf
import numpy as np

radar_gen = RadarGenV2(fs=100, mtime=20)
signal_hb_res = radar_gen.i_signal()
res_signal=radar_gen.gen_respirator_v1( kb=0.01, ti=2, te=2, tc=0.8)
hb_signal = radar_gen.gen_heartbeat_movement_signal_v1(am=0.05 * 10 * pow(10, -3), freq=1.2, phase=0)
filtered_signal = btwf.butter_bandpass_filter(data=signal_hb_res, fs=radar_gen.fs, order=4)

# normalization
filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
peaks = btwf.find_hr(data=signal_hb_res, fs=radar_gen.fs, order=4, distance=50, prominence=1)
print(len(peaks)*3)
time_arr_resp = np.arange(0, len(signal_hb_res), 1)
fig = px.line(x=time_arr_resp, y=filtered_signal, title="test")
fig.add_scatter(x=peaks, y=filtered_signal[peaks], mode='markers')
fig.show()

fig2 = px.line(x=time_arr_resp, y=signal_hb_res, title="test")
fig2.add_scatter(x=time_arr_resp, y=hb_signal, mode='lines')
fig2.show()

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