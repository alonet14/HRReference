import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import math
# ===config res_function parameter===
Kb = 10 * pow(10, -3)
Ti = 2
Te = 3
T = Ti + Te

t_c = 0.8
time_arr = np.arange(0, T+60, 0.01)

data_res = []
res_function_model = None

for t in time_arr:
    # time in period
    tip = t - T * int(t / T)
    # print(tip)
    if tip >= 0 and tip <= Ti:
        res_function_model = (-Kb / (Ti * Te)) * tip *tip + (Kb * T / (Ti * Te)) * tip
        # res_function_model = 0
    elif tip > Ti and tip <= T:
        res_function_model = (Kb / (1 - np.exp(-Te / t_c))) * tip *tip * (
                    np.exp(-(tip - Te) / t_c) - np.exp(-Te / t_c))

    data_res.append(res_function_model)


time_arr_resp=np.arange(0, len(data_res), 1)
fig=px.scatter(x=time_arr_resp, y=data_res)
fig.show()

# ======hb=====
data_hb=[]
disp_hb=0.5*pow(10, -3)
fhr=120
for t in time_arr:
    hb_function_model=disp_hb*np.sin(2*np.pi*fhr*t)
    data_hb.append(hb_function_model)

# fig = plt.plot(time_arr, data_res)
#
# plt.show()
