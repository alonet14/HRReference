import scipy.io
import plotly.express as px
import numpy as np
import utils_v1.butterworth_filter as btwf
parent_path='E:\\lab_project\\hr_processing\\data\\datasets'
file_name=parent_path+'\\measurement_data_person6\\PCG_front_radar_front\\PCG_2L_radar_5L\\DATASET_2017-02-06_09-01-56_Person 6.mat'
mat=scipy.io.loadmat(file_name)
i_signal=mat['radar_I']
i_signal = np.asarray(i_signal)
data=[]
for index, val in enumerate(i_signal):
    data.append(val[0])
data=data[3446:]
time_serries=np.arange(0, len(data)/500, 1/500)
# fig=px.line(x=time_serries, y=data, title='i signal')

peaks=btwf.find_hr(data, fs=500)
filtered_signal=btwf.butter_bandpass_filter(data, fs=500)
filtered_signal=filtered_signal/np.max(np.abs(filtered_signal))
fig=px.line(x=time_serries, y=filtered_signal)

# fig.add_scatter(x=time_serries, y=data/np.max(np.abs(data)))
fig.add_scatter(x=peaks/500, y=filtered_signal[peaks], mode='markers')
fig.show()

# fig2=px.line(x=time_serries, y=data, title='raw data')
# fig2.show()

