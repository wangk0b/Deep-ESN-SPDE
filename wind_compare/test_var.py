import numpy as np
import torch
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

device = torch.device("cuda:0")

wind = np.load("Lorenz96.npy")
#wind = np.load("sim_data.npy")

model = VAR(wind[0:800,:])
results = model.fit(1)

coef = np.array(results.params)
wind = torch.tensor(wind,dtype = torch.float32)
coef = torch.tensor(coef,dtype = torch.float32)
coef = coef.to(device)

num_pred = 3
in_mat = torch.cat((torch.ones(199,1),wind[800:999,:]),dim = 1)
in_mat = in_mat.to(device)
result = []
for i in range(num_pred):
    if i == 0: 
       predict = torch.mm(in_mat,coef)
    else:
        in_mat = torch.cat((torch.ones(199,1).to(device),predict),dim = 1)
        predict = torch.mm(in_mat,coef)
    result.append(torch.mean((predict[0:(199-i),:].cpu() - wind[(801+i):,:])**2).numpy())

with open('VAR_MSPE_96.txt','a') as f:
    for i in result:
        f.write(str(i)+" ")
    f.write('\n')
