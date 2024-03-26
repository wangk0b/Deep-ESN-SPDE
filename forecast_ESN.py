#import libraries

import netCDF4
import numpy as np
import sys
from time import perf_counter
import torch
#read in windresiudals data
m = int(sys.argv[1])
n_h = int(sys.argv[2])
l = float(sys.argv[3])
delta = float(sys.argv[4])
phi = float(sys.argv[5])
a_w = float(sys.argv[6])
a_u = float(sys.argv[7])
pi_w = float(sys.argv[9])
pi_u = float(sys.argv[9])
bat = int(sys.argv[10])

 ncin = netCDF4.Dataset('wind_residual_all_locations.nc', 'r', format='NETCDF4')
 wind_residual = ncin.variables['wind_residual_all_locations'][:]
 print(wind_residual.shape)
 ncin.close()
 loc = np.loadtxt("locselectR.txt", usecols=(0), delimiter=",", dtype=np.int32)-1
 wind = torch.tensor(wind_residual[loc,:],dtype = torch.float32)
 print(wind.size())

from ESN_Batch import ESN
from data import Data
from index import Index
#num_loc = int(sys.argv[11])
#loc = np.random.randint(0,3173,num_loc)
parameter=[m,n_h,l,delta,phi,a_w,a_u,pi_w,pi_u,bat]
print(wind.shape)
data=Data(35040,wind)
index=Index()
index.test_start=26280
index.test_end=35039
device = torch.device("cuda:0")
esn_model=ESN(data,index,device)
esn_model.train(parameter)
start = perf_counter()
esn_model.forecast()
stop = perf_counter()
print("The forecasting time is: ", stop - start)
esn_model.compute_forecast_mean()
esn_model.compute_forecast_error()
#MSPE = esn_model.compute_MSPE()
#MSPE = MSPE.numpy()
#print(MSPE.shape)
#file_name = "MSPE"+str(bat)+".npy"
#np.save(file_name,MSPE)
#print("Done!!!!")
esn_model.persistence_compute()


# tau=1 #set parameters
# m=1
# numLocs=20
# hyperpara = Hyperpara()
# nTime=wind_residual.shape[1]
# data=Data(nTime,wind_residual)
# #try a toy sample
# data_new=Data(30,wind_residual[0:20,0:30])
# index=Index()
# # first 90 as training and last 10 as testing
# index.test_start=20
# index.test_end=29
# #calculating training length
# inSampleEmb_len = index.test_start - m*tau
#
#
# #create X matrix
# #initialize
# inSampleX = np.repeat(np.nan,inSampleEmb_len *m * numLocs).reshape(inSampleEmb_len, m,-1)
# #fill the design matrix X
# #fill it with lag 1
# for i in range(inSampleEmb_len):
#    inSampleX[i,] = data_new.ts[range(i,(m * tau + i), tau)]
#
# #take the mean vector acrross all locations and standard deviation vector
#
# inSampleX_mean = inSampleX.mean(axis=0)
# inSampleX_std = inSampleX.std(axis=0)
# print(inSampleX_mean,"and\n",inSampleX_std)
# #standardize sample
# inSampleX = (inSampleX -inSampleX_mean)/inSampleX_std
#
# #form the design matrix
# inSampleDesignMatrix = np.column_stack([np.repeat(1,inSampleEmb_len),inSampleX.reshape(inSampleEmb_len,-1)])
# #Same procedure on the response variable Y
# inSampleY = data_new.ts[range(m * tau,inSampleEmb_len + (m * tau))]
# inSampleY_mean=inSampleY.mean(axis=0)
# inSampleY_std=inSampleY.std(axis=0)
# inSampleY = (inSampleY-inSampleY_mean)/inSampleY_std
# #now we have the response variable Y and Design matrix X
#
# #Now we do the same arrangements on the testing set
#
# #get indices for the testing set
# outSampleEmb_index = np.arange(index.test_start, index.test_end+1)
# print(inSampleEmb_len)
# #getting the length of testing set
# outSampleEmb_len = len(outSampleEmb_index)
#
# #create outsample design matrix
# outSampleX = np.zeros((outSampleEmb_len, m, numLocs)) * np.nan
# for i,ind in enumerate(outSampleEmb_index):
#    outSampleX[i,] = data_new.ts[range(ind - tau * m,ind, tau)]
# outSampleX = (outSampleX - inSampleX_mean)/inSampleX_std
# outSampleDesignMatrix=np.column_stack([np.repeat(1,outSampleEmb_len),outSampleX .reshape(outSampleEmb_len,-1)])
#
# #specify parameters to generate W and U
# nh=20
# wWidth=0.1
# uWidth=0.1
# wSparsity=0.3
# nColsU=numLocs*m+1
# uSparsity=0.3
# delta=1
# alpha=0.2
#
# wMat = np.random.uniform(-wWidth,wWidth,nh*nh).reshape(nh,-1)
# uMat = np.random.uniform(-uWidth,uWidth,nh*nColsU).reshape(nColsU,-1)
#
# for i in range(nh):
#    numReset=nh-np.random.binomial(nh,wSparsity)
#    resetIndex = np.random.choice(nh, numReset, replace = False)
#    wMat[resetIndex,i]=0
#
# for i in range(nColsU):
#    numReset=nh-np.random.binomial(nh,uSparsity)
#    resetIndex = np.random.choice(nh, numReset, replace = False)
#    uMat[i,resetIndex]=0
# v = eigh(wMat,eigvals_only=True)
# spectralRadius = max(abs(v))
# wMatScaled=wMat*delta/spectralRadius
# hMatDim = 2*nh
# uProdMat=inSampleDesignMatrix.dot(uMat)
# hMat = np.zeros((hMatDim,inSampleEmb_len))
# xTemp = uProdMat[0,:]
# xTemp = np.tanh(xTemp)
# hMat[0:nh,0] = xTemp
# hMat[nh:,0] = xTemp*xTemp
#
# for t in range(1,inSampleEmb_len):
#    xTemp = wMat.dot(xTemp)+uProdMat[t,:]
#    xTemp = np.tanh(xTemp)
#    hMat[0:nh,t] = xTemp*alpha + hMat[0:nh,t-1]*(1-alpha)
#    hMat[nh:,t] = hMat[0:nh,t]*hMat[0:nh,t]
#
# uProdMatOutSample = outSampleDesignMatrix.dot(uMat)
# hMatOutSample = np.zeros((outSampleEmb_len,hMatDim))
#
# xTemp = wMat.dot(xTemp)+uProdMatOutSample[0,:]
# xTemp = np.tanh(xTemp)
#
# hMatOutSample[0,0:nh] = xTemp
# hMatOutSample[0,nh:] = xTemp*xTemp
#
# for t in range(1,outSampleEmb_len):
#       xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
#       xTemp = np.tanh(xTemp)
#       hMatOutSample[t,0:nh] = xTemp*alpha + hMatOutSample[t-1,0:nh]*(1-alpha)
#       hMatOutSample[t,nh:] = hMatOutSample[t,0:nh]*hMatOutSample[t,0:nh]
#
# #finished creating all Hs
#
# #stat forcasting
# ensembleLen=1
# numTimePred=3
# ridge=0.2
#
# forMat = np.ones((ensembleLen,outSampleEmb_len,numLocs,numTimePred)) * np.nan
# tmp = hMat.dot(hMat.transpose())
# np.fill_diagonal(tmp,tmp.diagonal()+ridge)
# forMat[0,:,:,0] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(inSampleY)))
# forMat[0,:,:,0] = forMat[0,:,:,0] * inSampleY_std + inSampleY_mean
# hMatDim = 2*nh
# outSampleX_mixed = outSampleX.copy()
#
# for pred_lag in range(1,numTimePred):
#     #Create H Matrix out-sample for prediction more than one lead time
#     outSampleX_mixed = outSampleX.copy()
#     for i in range(min(pred_lag,m)):
#         ii = i+1
#         outSampleX_mixed[pred_lag:,-ii,:] = (forMat[0,(pred_lag-ii):(-ii),:,pred_lag-ii] -
#                                                               inSampleX_mean[-ii])/inSampleX_std[-ii]
#
#     outSampleX_mixed[0:pred_lag,] = np.nan
#     outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,outSampleEmb_len),
#                                                     outSampleX_mixed.reshape(outSampleEmb_len,-1)])
#     uProdMatOutSample = outSampleDesignMatrix_mixed.dot(uMat)
#
#     hMatOutSample_new = np.zeros((outSampleEmb_len,hMatDim)) * np.nan
#
#     xTemp = hMatOutSample[pred_lag-1,0:nh]
#     xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
#     xTemp = np.tanh(xTemp)
#
#     hMatOutSample_new[pred_lag,0:nh] = xTemp
#     hMatOutSample_new[pred_lag,nh:] = xTemp*xTemp
#
#     for t in range(pred_lag+1,outSampleEmb_len):
#         xTemp = hMatOutSample[t-1,0:nh]
#         xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
#         xTemp = np.tanh(xTemp)
#
#         hMatOutSample_new[t,0:nh] = xTemp*alpha + hMatOutSample_new[t-1,0:nh]*(1-alpha)
#         hMatOutSample_new[t,nh:] = hMatOutSample_new[t,0:nh] * hMatOutSample_new[t,0:nh]
#     hMatOutSample = hMatOutSample_new.copy()
#
#     forMat[0,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(inSampleY)))
#
#                 #Transform to the original scale
#     forMat[0,:,:,pred_lag] = forMat[0,:,:,pred_lag] * inSampleY_std + inSampleY_mean
#     print(forMat[0,:,:,1])
#     forMean = forMat.mean(axis=0)

