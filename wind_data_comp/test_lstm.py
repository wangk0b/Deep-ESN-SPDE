from LSTM_GRU import GRUNet, train, evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader

import netCDF4
ncin = netCDF4.Dataset('../wind_residual_all_locations.nc', 'r', format='NETCDF4')
wind_residual = ncin.variables['wind_residual_all_locations'][:]
ncin.close()
loc = np.loadtxt("../locselectR.txt", usecols=(0), delimiter=",", dtype=np.int32)-1
wind = torch.tensor(wind_residual[loc,:].T,dtype = torch.float32)


#wind = np.load("Lorenz96.npy")

#wind = np.load("../wind_sim/sim_data.npy")
num_pred = 3
d = torch.device("cuda:0")
result = []
for epoch in [1,20,30,40,50]:
 for i in range(num_pred):
   x_train = torch.tensor(wind[0:(26279-i),:],dtype=torch.float32)
   y_train = torch.tensor(wind[(i+1):26280,:], dtype = torch.float32)
   if i == 0:
    x_test = torch.tensor(wind[26280:(35039-i),:],dtype = torch.float32)
   y_test = torch.tensor(wind[(26281+i):35040,:],dtype = torch.float32)


   train_loader = DataLoader(list(zip(x_train,y_train)), shuffle=False, batch_size=1)

   test_loader = DataLoader(list(zip(x_test,y_test)), shuffle=False, batch_size=1)


   lr = 0.001
   print("Current lead is {}".format(i+1))
   if i == 0:
     gru_model = train(train_loader, lr, 100, epoch, d, model_type="LSTM")
   x_test, _, gru_sMAPE = evaluate(gru_model,test_loader, d)
   if epoch == 1 and i == 0:
       np.save("LSTM_predict.npy",x_test)
   x_test = torch.tensor(x_test,dtype = torch.float32)
   result.append(gru_sMAPE)

#with open("LSTM_MSPE_WIND.txt",'a') as f:
#    for i in result:
#        f.write(str(i)+" ")
#    f.write('\n')



