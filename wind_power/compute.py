import netCDF4
import numpy as np
nTime = 35040

ncin = netCDF4.Dataset('../wind_residual_all_locations.nc', 'r', format='NETCDF4')
gamma_all_locations = ncin.variables['gamma'][:] # scaling parameter gamma(s) at all 53333 locations
wind_residual_all_locations = ncin.variables['wind_residual_all_locations'][:]
harmonics_coefficients_all_locations = ncin.variables['harmonics_coefficients'][:]  # armonics_coefficients at all 53333 locations
harmonics = ncin.variables['harmonics'][:] # harmonics basis functions for each hour from 2013 to 2016
ncin.close()

ncin = netCDF4.Dataset('../wind_farm_data.nc', 'r', format='NETCDF4')
alpha_turbine_location = ncin.variables['alpha_turbine_location'][:] # alpha in the wind power law for 75 wind farms at each hour of the day
index_turbine_location = ncin.variables['index_turbine_location'][:]  # index of the 75 wind farms locations in the all 53333 locations
turbine_height = ncin.variables['turbine_height'][:] # turbine heights in the 75 wind farms
turbine_type = ncin.variables['turbine_type'][:] # turbine types in the 75 wind farms
ncin.close()

index_turbine_location = index_turbine_location.astype('int')
turbine_type = turbine_type.astype('int')



harmonic_mean_turbine_locations = harmonics[(nTime - 365*24):].dot(harmonics_coefficients_all_locations[index_turbine_location,1:].T)
harmonic_mean_turbine_locations += harmonics_coefficients_all_locations[index_turbine_location,0]

true_wind_speed_turbine_location = harmonic_mean_turbine_locations + wind_residual_all_locations[index_turbine_location,(nTime - 365*24):].T * gamma_all_locations[index_turbine_location]
true_wind_speed_turbine_location = true_wind_speed_turbine_location.T **2

del wind_residual_all_locations
# other models

index_s = np.loadtxt("../locselectR.txt",dtype = np.int32, usecols = (0),delimiter = ",") - 1
index_all = np.arange(53333)
index_p = list(set(index_all) - set(index_s))

#LSTM
LSTM_all = np.zeros((8760,53333))*np.nan
LSTM_s = np.load("../wind_data_comp/LSTM_predict.npy")
LSTM_m = np.load("../LSTM_SPDE.npy")


LSTM_all[1:8750,index_s] = LSTM_s[1:8750,:]
LSTM_all[1:8750,index_p] = LSTM_m[1:8750,:]
del LSTM_s
del LSTM_m


LSTM_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + LSTM_all[:,index_turbine_location] * gamma_all_locations[index_turbine_location]
LSTM_wind_speed_forecast_turbine_location = LSTM_wind_speed_forecast_turbine_location**2
del LSTM_all
#GRU 
GRU_all = np.zeros((8760,53333))*np.nan
GRU_s = np.load("../wind_data_comp/GRU_predict.npy")
GRU_m = np.load("../GRU_SPDE.npy")


GRU_all[1:8750,index_s] = GRU_s[1:8750,:]
GRU_all[1:8750,index_p] = GRU_m[1:8750,:]
del GRU_s
del GRU_m


GRU_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + GRU_all[:,index_turbine_location] * gamma_all_locations[index_turbine_location]
GRU_wind_speed_forecast_turbine_location = GRU_wind_speed_forecast_turbine_location**2
del GRU_all




#VAR
VAR_all = np.zeros((8760,53333))*np.nan
VAR_s = np.load("../wind_data_comp/VAR_predict.npy")
VAR_m = np.load("../VAR_SPDE.npy")


VAR_all[1:8750,index_s] = VAR_s[1:8750,:]
VAR_all[1:8750,index_p] = VAR_m[1:8750,:]
del VAR_s
del VAR_m


VAR_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + VAR_all[:,index_turbine_location] * gamma_all_locations[index_turbine_location]
VAR_wind_speed_forecast_turbine_location = VAR_wind_speed_forecast_turbine_location**2
del VAR_all


#ESN
ESN_all = np.zeros((8760,53333))*np.nan
ESN_s = np.load("../ESN_ensembles.npy")[:,:,:,1].mean(axis = 0)
ESN_m = np.load("ESN_mean_predict.npy")


ESN_all[:,index_s] = ESN_s
ESN_all[1:8750,index_p] = ESN_m/100
del ESN_s
del ESN_m

ESN_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations +ESN_all[:,index_turbine_location] * gamma_all_locations[index_turbine_location]
ESN_wind_speed_forecast_turbine_location = ESN_wind_speed_forecast_turbine_location**2
del ESN_all


#LAT
lat_all = np.zeros((8760,53333))*np.nan
lat_s = np.load("../ESN_ensembles.npy")[:,:,:,1].mean(axis = 0)
lat_m = np.load("../LatKrig_Pred2.npy").reshape(8749,50160)


lat_all[:,index_s] = lat_s
lat_all[1:8750,index_p] = lat_m
del lat_s
del lat_m


lat_wind_speed_forecast_turbine_location = harmonic_mean_turbine_locations + lat_all[:,index_turbine_location] * gamma_all_locations[index_turbine_location]
lat_wind_speed_forecast_turbine_location = lat_wind_speed_forecast_turbine_location**2
del lat_all








#######compute 

wind_speed = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37])

power_curve = np.array([[0,0,0,0,0,0,33,106,197,311,447,610,804,1032,1298,1601,1936,2292,2635,2901,3091,3215,3281,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,29,68,114,177,243,347,452,595,738,907,1076,1307,1538,1786,2033,2219,2405,2535,2633,2710,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


def get_power_curve(t_m,x):
    if x > 37:
        return 0
    left_ind = int(x*2)
    power = (power_curve[t_m,left_ind+1]-power_curve[t_m,left_ind]) * (x - wind_speed[left_ind])/0.5 + power_curve[t_m,left_ind]
    return(power)

#get true power curve
true_power_turbine_height = np.empty(true_wind_speed_turbine_location.shape) * np.nan
for loc in np.arange(75):
    for time in np.arange(1,8750):
        tmp =  true_wind_speed_turbine_location[loc,time]*(turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
        true_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)



esn_power_turbine_height = np.empty((75,8750)) * np.nan
lat_power_turbine_height = np.empty((75,8750)) * np.nan
LSTM_power_turbine_height = np.empty((75,8750)) * np.nan
GRU_power_turbine_height = np.empty((75,8750)) * np.nan
VAR_power_turbine_height = np.empty((75,8750)) * np.nan

#forcasted power curver

for loc in np.arange(75):
        for time in np.arange(1,8750):
            tmp = ESN_wind_speed_forecast_turbine_location[time,loc] *(turbine_height[loc]/10)**alpha_turbine_location[loc,time%24] 
            esn_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp) 
            tmp = GRU_wind_speed_forecast_turbine_location[time,loc] *(turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
            GRU_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp) 
            tmp = LSTM_wind_speed_forecast_turbine_location[time,loc] * (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
            LSTM_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp) 
            tmp = VAR_wind_speed_forecast_turbine_location[time,loc] * (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
            VAR_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)
            tmp = lat_wind_speed_forecast_turbine_location[time,loc] * (turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
            lat_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)




esn_wind_energy_diff   = np.abs(esn_power_turbine_height - true_power_turbine_height[:,:8750])
lat_wind_energy_diff   = np.abs(lat_power_turbine_height - true_power_turbine_height[:,:8750])
VAR_wind_energy_diff   = np.abs(VAR_power_turbine_height - true_power_turbine_height[:,:8750])
GRU_wind_energy_diff   = np.abs(GRU_power_turbine_height - true_power_turbine_height[:,:8750])
LSTM_wind_energy_diff   = np.abs(LSTM_power_turbine_height - true_power_turbine_height[:,:8750])



np.save("esn_wind_energy.npy",esn_wind_energy_diff)
np.save("lat_wind_energy.npy",lat_wind_energy_diff)
np.save("VAR_wind_energy.npy",VAR_wind_energy_diff)
np.save("LSTM_wind_energy.npy",LSTM_wind_energy_diff)
np.save("GRU_wind_energy.npy",GRU_wind_energy_diff)

print("success!!!")
