from LSTM_GRU import GRUNet, train, evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
wind = np.load("Lorenz96.npy")
#wind = np.load("../wind_sim/sim_data.npy")
num_pred = 3
d = torch.device("cuda:0")
result = []
for epoch in [150,200]:
 for i in range(num_pred):
   x_train = torch.tensor(wind[0:(799-i),:],dtype=torch.float32)
   y_train = torch.tensor(wind[(i+1):800,:], dtype = torch.float32)
   if i == 0:
    x_test = torch.tensor(wind[800:(999-i),:],dtype = torch.float32)
   y_test = torch.tensor(wind[(801+i):1000,:],dtype = torch.float32)


   train_loader = DataLoader(list(zip(x_train,y_train)), shuffle=False, batch_size=1)

   test_loader = DataLoader(list(zip(x_test,y_test)), shuffle=False, batch_size=1)


   lr = 0.001
   print("Current lead is {}".format(i+1))
   if i == 0:
     gru_model = train(train_loader, lr, 80, epoch, d, model_type="GRU")
   x_test, _, gru_sMAPE = evaluate(gru_model,test_loader, d)
   x_test = torch.tensor(x_test,dtype = torch.float32)
   result.append(gru_sMAPE)

with open('GRU_MSPE_96.txt','a') as f:
    for i in result:
        f.write(str(i)+" ")
    f.write('\n')

