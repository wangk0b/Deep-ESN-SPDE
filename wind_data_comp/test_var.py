import numpy as np
import torch
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


import netCDF4
ncin = netCDF4.Dataset('../wind_residual_all_locations.nc', 'r', format='NETCDF4')
wind_residual = ncin.variables['wind_residual_all_locations'][:]
ncin.close()
loc = np.loadtxt("../locselectR.txt", usecols=(0), delimiter=",", dtype=np.int32)-1
wind = wind_residual[loc,:].T



device = torch.device("cuda:0")

#wind = np.load("Lorenz96.npy")
#wind = np.load("sim_data.npy")

model = VAR(wind[0:26280,:])
results = model.fit(1)

coef = np.array(results.params)
wind = torch.tensor(wind,dtype = torch.float32)
coef = torch.tensor(coef,dtype = torch.float32)
coef = coef.to(device)

num_pred = 3
in_mat = torch.cat((torch.ones(8759,1),wind[26280:35039,:]),dim = 1)
in_mat = in_mat.to(device)
result = []
for i in range(num_pred):
    if i == 0: 
       predict = torch.mm(in_mat,coef)
       #np.save("VAR_predict.npy",predict)
    else:
        in_mat = torch.cat((torch.ones(8759,1).to(device),predict),dim = 1)
        predict = torch.mm(in_mat,coef)
        if i==1:
            np.save("VAR_predict.npy",predict.to('cpu').detach().numpy())
    result.append(torch.mean((predict[0:(8759-i),:].cpu() - wind[(26281+i):,:])**2).numpy())

#with open('VAR_MSPE_WIND.txt','a') as f:
#    for i in result:
#        f.write(str(i)+" ")
#    f.write('\n')
