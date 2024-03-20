import netCDF4
import numpy as np
nTime = 35040

wind_speed = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,35,35.5,36,36.5,37])

power_curve = np.array([[0,0,0,0,0,0,33,106,197,311,447,610,804,1032,1298,1601,1936,2292,2635,2901,3091,3215,3281,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,3300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,29,68,114,177,243,347,452,595,738,907,1076,1307,1538,1786,2033,2219,2405,2535,2633,2710,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,2750,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

def get_power_curve(t_m,x):
    if x > 37:
       return 0
    left_ind = int(x*2)
    power = (power_curve[t_m,left_ind+1]-power_curve[t_m,left_ind]) * (x - wind_speed[left_ind])/0.5 + power_curve[t_m,left_ind]
    return(power)






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



true_power_turbine_height = np.empty(true_wind_speed_turbine_location.shape) * np.nan
for loc in np.arange(75):
    for time in np.arange(1,8750):
        tmp =  true_wind_speed_turbine_location[loc,time]*(turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
        true_power_turbine_height[loc,time] = get_power_curve(turbine_type[loc],tmp)




#del wind_residual_all_locations
# other models

index_s = np.loadtxt("../locselectR.txt",dtype = np.int32, usecols = (0),delimiter = ",") - 1
index_all = np.arange(53333)
index_p = list(set(index_all) - set(index_s))

#ESN
ESN_all = np.zeros((8760,53333,3))*np.nan
ESN_s = np.load("../ESN_ensembles.npy")[:,:,:,:].mean(axis = 0)
ESN_m2 = np.load("ESN_mean_predict.npy")
ESN_m1 = np.load("ESN_mean_predictone.npy")
ESN_m3 = np.load("ESN_mean_predictthree.npy")

ESN_all[:,index_s,:] = ESN_s
ESN_all[:8750,index_p,0] = ESN_m1
ESN_all[1:8750,index_p,1] = ESN_m2/100
ESN_all[2:8750,index_p,2] = ESN_m3

del ESN_s
del ESN_m1
del ESN_m2
del ESN_m3


forMean2016_all = ESN_all
del ESN_all

quantile = np.arange(0.025,1,0.025)


lower_ind = np.empty(3).astype('int')
upper_ind = np.empty(3).astype('int')

# 95%
lower_ind[0] = np.where( np.abs(quantile - 0.025) < 1e-12 ) [0][0]
upper_ind[0] = np.where( np.abs(quantile - 0.975) < 1e-12 )[0][0]

# 80%
lower_ind[1] = np.where( np.abs(quantile - 0.1) < 1e-12 ) [0][0]
upper_ind[1] = np.where( np.abs(quantile - 0.9) < 1e-12 )[0][0]

# 60%
lower_ind[2] = np.where( np.abs(quantile - 0.2) < 1e-12 ) [0][0]
upper_ind[2] = np.where( np.abs(quantile - 0.8) < 1e-12 )[0][0]


df = np.load('quantiles_all.npz')
forErrorQuantile_all = df['forErrorQuantile_all']

outSampleEmb_index = np.arange(26280, 35040)
forecast = forMean2016_all
true = wind_residual_all_locations[:,outSampleEmb_index].T
PI_all = np.empty((3,3,53333))

for i in range(3):
    LB = forecast + forErrorQuantile_all[lower_ind[i]]
    UB = forecast + forErrorQuantile_all[upper_ind[i]]

    for j in range(3):
        PI_all[i,j] = (np.logical_and(true < UB[:,:,j],true > LB[:,:,j])).sum(axis = 0) / (~np.isnan(UB[:,:,j])).sum(axis = 0)

PI_all_mean = np.mean(PI_all, axis = 2)
PI_all_std = np.std(PI_all, axis = 2)


esn_turbine_location = forMean2016_all[:,index_turbine_location,1]
esn_turbine_location_quantile = np.empty((quantile.size, *esn_turbine_location.shape))

for i in range(quantile.size):
    esn_turbine_location_quantile[i] = esn_turbine_location + forErrorQuantile_all[i,index_turbine_location,1]

esn_wind_speed_turbine_location_quantile = harmonic_mean_turbine_locations + esn_turbine_location_quantile * gamma_all_locations[index_turbine_location]

esn_wind_speed_turbine_location_quantile = esn_wind_speed_turbine_location_quantile ** 2

esn_power_turbine_height_quantile   = np.zeros_like(esn_wind_speed_turbine_location_quantile)

for i in range(quantile.size):
        for loc in range(75):
            for time in np.arange(1,8750): 

                 tmp = esn_wind_speed_turbine_location_quantile[i,time,loc]*(turbine_height[loc]/10)**alpha_turbine_location[loc,time%24]
                 esn_power_turbine_height_quantile[i,time,loc] = get_power_curve(turbine_type[loc],tmp) 

esn_total_power_turbine_height_quantile_diff = np.nansum(np.abs(esn_power_turbine_height_quantile - true_power_turbine_height.T), axis = (1,2))
np.savez(file = 'power_quantiles.npz', esn_power_turbine_height_quantile = esn_total_power_turbine_height_quantile_diff)














print("success!!!")
