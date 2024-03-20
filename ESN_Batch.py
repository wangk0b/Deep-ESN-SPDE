# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from time import perf_counter

import torch
import numpy as np
# from scipy.linalg import eigh
# from sklearn.covariance import GraphicalLassoCV

class ESN:

    def __init__(self,data,index,device):
        self.data = data
        self.index = index

        self.ensembleLen = 100
        self.numTimePred = 3

        self.tauEmb = 1 # number of lead time embedded

        self.forMeanComputed = False
        self.forErrorComputed = False

       # self.batch = 3 # batch size

        self.device = device
        self.dtype = torch.float32
        print("The device used is ",self.device)

    def standardize_in_sample(self, is_validation = False):
        if(is_validation):
            self.inSampleEmb_len = self.index.validate_start - self.m *self.tauEmb
        else:
            self.inSampleEmb_len = self.index.test_start - self.m *self.tauEmb

        #### X
        self.inSampleX = torch.zeros((self.inSampleEmb_len, self.m, self.numLocs), device=self.device)*torch.nan
        for i in range(self.inSampleEmb_len):
            self.inSampleX[i,] = self.data.ts[range(i,(self.m * self.tauEmb + i), self.tauEmb)]

        self.inSampleX_mean = self.inSampleX.mean(axis=0)
        self.inSampleX_std = self.inSampleX.std(axis=0)

        self.inSampleX = (self.inSampleX -self.inSampleX_mean) / self.inSampleX_std
        self.inSampleDesignMatrix = torch.column_stack([torch.ones(self.inSampleEmb_len, dtype = torch.int8, device = self.device), self.inSampleX.reshape(self.inSampleEmb_len,-1)])

        #### Y
        self.inSampleY = self.data.ts[range(self.m * self.tauEmb,self.inSampleEmb_len + (self.m * self.tauEmb))].float()

        self.inSampleY_mean = self.inSampleY.mean(axis=0)
        self.inSampleY_std=self.inSampleY.std(axis=0)

        self.inSampleY = (self.inSampleY-self.inSampleY_mean)/self.inSampleY_std

    def standardize_out_sample(self, is_validation = False):
        if(is_validation):
            self.outSampleEmb_index = np.arange(self.index.validate_start, self.index.validate_end+1)
        else:
            self.outSampleEmb_index = np.arange(self.index.test_start, self.index.test_end+1)

        self.outSampleEmb_len = len(self.outSampleEmb_index)
        #### X
        self.outSampleX = torch.zeros((self.outSampleEmb_len, self.m, self.numLocs), device = self.device) * torch.nan
        for i,ind in enumerate(self.outSampleEmb_index):
            self.outSampleX[i,] = self.data.ts[range(ind - self.tauEmb * self.m,ind, self.tauEmb)]
        self.outSampleX = (self.outSampleX - self.inSampleX_mean)/self.inSampleX_std
        self.outSampleDesignMatrix = torch.column_stack([torch.ones(self.outSampleEmb_len, dtype = torch.int8, device = self.device), self.outSampleX.reshape(self.outSampleEmb_len,-1)])

        #### Y
        self.outSampleY = (self.data.ts[self.outSampleEmb_index] - self.inSampleY_mean)/self.inSampleY_std

    def get_w_and_u(self):
        wMat = torch.FloatTensor(self.nh*self.nh).to(self.device).uniform_(-self.wWidth,self.wWidth).reshape(self.nh, -1)
        uMat = torch.FloatTensor(self.nh*self.nColsU).to(self.device).uniform_(-self.uWidth,self.uWidth).reshape(self.nColsU, -1)

        #Make W Matrix Sparse
        for i in range(self.nh):
            numReset=self.nh-np.random.binomial(self.nh,self.wSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            wMat[resetIndex,i]=0

        #Make U Matrix Sparse
        for i in range(self.nColsU):
            numReset=self.nh-np.random.binomial(self.nh,self.uSparsity)
            resetIndex = np.random.choice(self.nh, numReset, replace = False)
            uMat[i,resetIndex]=0

        #Scale W Matrix
        v = torch.linalg.eigvalsh(wMat)
        spectralRadius = max(abs(v))
        wMatScaled=wMat*self.delta/spectralRadius

        return wMatScaled, uMat

    def get_hMat(self,wMat,uMat):
        #Create H Matrix in-sample
        hMatDim = 2*self.nh
        uProdMat = torch.mm(self.inSampleDesignMatrix, uMat)

        hMat = torch.zeros((hMatDim,self.inSampleEmb_len), device = self.device)

        xTemp = uProdMat[0,:]
        xTemp = torch.tanh(xTemp)

        hMat[0:self.nh,0] = xTemp
        hMat[self.nh:,0] = xTemp*xTemp

        for t in range(1,self.inSampleEmb_len):
            xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMat[t,:]
            xTemp = torch.tanh(xTemp)

            hMat[0:self.nh,t] = xTemp*self.alpha + hMat[0:self.nh,t-1]*(1-self.alpha)
            hMat[self.nh:,t] = hMat[0:self.nh,t]*hMat[0:self.nh,t]

        #Create H Matrix out-sample
        uProdMatOutSample = torch.mm(self.outSampleDesignMatrix, uMat)
        hMatOutSample = torch.zeros((self.outSampleEmb_len,hMatDim), device = self.device)

        xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMatOutSample[0,:]
        xTemp = torch.tanh(xTemp)

        hMatOutSample[0,0:self.nh] = xTemp
        hMatOutSample[0,self.nh:] = xTemp*xTemp
        for t in range(1,self.outSampleEmb_len):
            xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMatOutSample[t,:]
            xTemp = torch.tanh(xTemp)

            hMatOutSample[t,0:self.nh] = xTemp*self.alpha + hMatOutSample[t-1,0:self.nh]*(1-self.alpha)
            hMatOutSample[t,self.nh:] = hMatOutSample[t,0:self.nh]*hMatOutSample[t,0:self.nh]

        del xTemp

        return hMat, hMatOutSample

    def train(self, hyper_para):

        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity,self.batch = hyper_para;


        self.numLocs = self.data.ts.shape[1]

        self.standardize_in_sample()
    def forecast(self):
        '''
            Forecast.

            Obtain the forecast matrix, forMat:
                * dimension: (#ensemble, #forecast time points, #locations, #prediction ahead time)
                * forMat[e,t,s,p] is the (p+1)-time ahead forecast for time t (instead of t+p+1!!)
                    at location s from e-th ensemble
        '''
        print("Forecasting, ensemble: ", end="")
        self.standardize_out_sample()
        self.forMat = torch.ones((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred), device = "cpu") * torch.nan
        self.nColsU = self.numLocs * self.m + 1
        training = torch.ones((self.ensembleLen,self.inSampleEmb_len,self.numLocs), device = "cpu") * torch.nan
        self.inSampleY_mean = self.inSampleY_mean.to(self.device)
        self.inSampleY_std = self.inSampleY_std.to(self.device)
        for iEnsem in range(self.ensembleLen):
            print(iEnsem+1,end=" ")

            forMat_iEnsem___0 = torch.zeros_like(self.forMat[iEnsem,:,:,0], device = self.device)

            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);

            tmp = torch.mm(hMat, hMat.T)
            tmp[range(len(tmp)), range(len(tmp))] = tmp.diagonal()+self.ridge
            BMat = torch.linalg.solve(tmp, torch.mm(hMat, self.inSampleY.to(self.device)))

            hMat_updated = hMat.clone()
            self.inSampleY_updated = self.inSampleY.clone()
            BMat_all = BMat.clone().to("cpu")

            # 1hour ahead predictions
            for k in range(int(self.outSampleEmb_len / self.batch)): # batch number = k+1
                hMatBatch = hMatOutSample[(k*self.batch):((k+1)*self.batch),]

                forMat_iEnsem___0 = torch.mm(hMatBatch, BMat)

                # forMat_iEnsem___0 = torch.mm(hMatOutSample, torch.linalg.solve(tmp, torch.mm(hMat, self.inSampleY)))

                forMat_iEnsem___0 = forMat_iEnsem___0*self.inSampleY_std + self.inSampleY_mean
                self.forMat[iEnsem,(k*self.batch):((k+1)*self.batch),:,0] = forMat_iEnsem___0.cpu()
                if k > 0:
                 hMat_updated = torch.column_stack([hMat_updated, hMatBatch.T])
                 tmp = torch.mm(hMat_updated, hMat_updated.T)
                 tmp[range(len(tmp)), range(len(tmp))] = tmp.diagonal() + self.ridge

                 self.inSampleY_updated = torch.row_stack([self.inSampleY_updated, self.outSampleY[(k*self.batch):((k+1)*self.batch),]])

                 BMat = torch.linalg.solve(tmp, torch.mm(hMat_updated, self.inSampleY_updated.to(self.device)))
                 BMat_all = torch.row_stack([BMat_all, BMat.to("cpu")])
            training[iEnsem,:,:] = torch.mm(hMat.t(),BMat)
            del BMat
            hMatDim = 2*self.nh
            #BMat_all = BMat_all[:hMatDim*(k+1),]

            #### Prediction at t + pred_lag + 1 where t is the current time
            for pred_lag in range(1,self.numTimePred):
                #Create H Matrix out-sample for prediction more than one lead time
                self.outSampleX_mixed = self.outSampleX.clone()
                for i in range(min(pred_lag, self.m)):
                    ii = i+1
                    self.outSampleX_mixed[pred_lag:,-ii,:] = (self.forMat[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii].to(self.device) - self.inSampleX_mean[-ii]) / self.inSampleX_std[-ii]

                self.outSampleX_mixed[0:pred_lag,] = torch.nan
                self.outSampleDesignMatrix_mixed = torch.column_stack([torch.ones(self.outSampleEmb_len, device = self.device),
                                                    self.outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])

                uProdMatOutSample = torch.mm(self.outSampleDesignMatrix_mixed, uMat)
                hMatOutSample_new = torch.zeros((self.outSampleEmb_len, hMatDim), device = self.device)*torch.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMatOutSample[pred_lag,:]
                xTemp = torch.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = torch.mm(wMat, xTemp.reshape(-1,1)).reshape(1,-1) + uProdMatOutSample[t,:]
                    xTemp = torch.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh]*hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.clone()

                for k in range(int(self.outSampleEmb_len / self.batch)): # batch number = k+1
                    hMatBatch = hMatOutSample[(k*self.batch):((k+1)*self.batch),:]
                    BMat = BMat_all[(k*hMatDim):((k+1)*hMatDim),:].to(self.device)
                    forMat_iEnsem___0 = torch.mm(hMatBatch, BMat)
                    forMat_iEnsem___0 = forMat_iEnsem___0*self.inSampleY_std + self.inSampleY_mean
                    self.forMat[iEnsem,(k*self.batch):((k+1)*self.batch),:,pred_lag] = forMat_iEnsem___0.cpu()
        training= training.mean(axis = 0)
        return training
          
    def cross_validation(self,cv_para,mChanged = True):
        '''
            Input:
                cv_para: the cross-validation parameter [m, nh, ridge, delta, alpha, wWidth, uWidth, wSparsity, uSparsity]
                mChange: if m in this cross-validation is different than the last one. If no, there is no need to
                        re-standardize the in-sample and out-sample data
            Output:
                MSE: vector of MSE with dimension self.numTimePred, which are the mean forecast square error for the different
                     time ahead forecast
        '''

        print("Cross Validation with Multiple Lead Times:")

        self.numLocs = self.data.ts.shape[1]

        self.m, self.nh, self.ridge, self.delta, self.alpha, self.wWidth, self.uWidth, self.wSparsity, self.uSparsity = cv_para;

        self.m = int(self.m)
        self.nh = int(self.nh)

        self.nColsU = self.numLocs * self.m + 1

        if(mChanged):
            self.standardize_in_sample(True)
            self.standardize_out_sample(True)

        forMatCV = np.zeros((self.ensembleLen,self.outSampleEmb_len,self.numLocs,self.numTimePred))

        for iEnsem in range(self.ensembleLen):
            wMat, uMat = self.get_w_and_u();

            hMat, hMatOutSample = self.get_hMat(wMat,uMat);

            #Ridge Regression to get out-sample forecast
            tmp = hMat.dot(hMat.transpose())
            np.fill_diagonal(tmp,tmp.diagonal()+self.ridge)

            forMatCV[iEnsem,:,:,0] += hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))

            #### Prediction at t + pred_lag + 1 where t is the current time
            hMatDim = 2*self.nh
            for pred_lag in range(1,self.numTimePred):

                #Create H Matrix out-sample for prediction more than one lead time
                outSampleX_mixed = self.outSampleX.copy()

                for i in range(min(pred_lag,self.m)):
                    ii = i+1
                    forMatCV_scaled_back  = forMatCV[iEnsem,(pred_lag-ii):(-ii),:,pred_lag-ii] * self.inSampleY_std + self.inSampleY_mean
                    outSampleX_mixed[pred_lag:,-ii,:] = (forMatCV_scaled_back - self.inSampleX_mean[-ii])/self.inSampleX_std[-ii]

                outSampleX_mixed[0:pred_lag,] = np.nan
                outSampleDesignMatrix_mixed = np.column_stack([np.repeat(1,self.outSampleEmb_len),
                                                    outSampleX_mixed.reshape(self.outSampleEmb_len,-1)])

                uProdMatOutSample = outSampleDesignMatrix_mixed.dot(uMat)

                hMatOutSample_new = np.zeros((self.outSampleEmb_len,hMatDim)) * np.nan

                xTemp = hMatOutSample[pred_lag-1,0:self.nh]
                xTemp = wMat.dot(xTemp)+uProdMatOutSample[pred_lag,:]
                xTemp = np.tanh(xTemp)

                hMatOutSample_new[pred_lag,0:self.nh] = xTemp
                hMatOutSample_new[pred_lag,self.nh:] = xTemp*xTemp

                for t in range(pred_lag+1,self.outSampleEmb_len):
                    xTemp = hMatOutSample[t-1,0:self.nh]
                    xTemp = wMat.dot(xTemp)+uProdMatOutSample[t,:]
                    xTemp = np.tanh(xTemp)

                    hMatOutSample_new[t,0:self.nh] = xTemp*self.alpha + hMatOutSample_new[t-1,0:self.nh]*(1-self.alpha)
                    hMatOutSample_new[t,self.nh:] = hMatOutSample_new[t,0:self.nh] * hMatOutSample_new[t,0:self.nh]

                hMatOutSample = hMatOutSample_new.copy()

                forMatCV[iEnsem,:,:,pred_lag] = hMatOutSample.dot(np.linalg.solve(tmp,hMat.dot(self.inSampleY)))


        forMatCVmean = forMatCV.mean(axis = 0)

        diff = np.ndarray(shape = forMatCVmean.shape) * np.nan

        for i in range(self.numTimePred):
            diff[:,:,i] = forMatCVmean[:,:,i] - self.outSampleY

        MSPE = np.nanmean(diff**2,axis=(0,1))
        forcast = forMatCVmean
        actual = self.outSampleY
        return MSPE, forcast, actual

    def compute_forecast_mean(self):
        '''
            Compute the ensemble forecast mean, forMean:
                * dimension: (#forecast time points, #locations, #prediction ahead time)
                * forMean[t,s,p] is the (p+1)-time ahead forecast mean for time t (instead of t+p+1!!) at location s
        '''
        self.forMean = self.forMat.mean(axis=0)
        self.forMeanComputed = True
        return self.forMean

    def compute_forecast_error(self):
        '''
            Compute the error by the ensemble forecast mean, forError:
                * dimension: (#forecast time points, #locations, #prediction ahead time)
                * forError[t,s,p] is the (p+1)-time ahead forecast error for time t (instead of t+p+1!!) at location s
        '''
        if(not self.forMeanComputed):
            self.compute_forecast_mean()

        # PYTORCH 35
        self.forError = torch.zeros_like(self.forMean)
        # Original
        # self.forError = np.zeros_like(self.forMean)
        # PYTORCH 36
        self.forError.fill_(torch.nan)
        # Original
        # self.forError.fill(np.nan)
        for ahead in range(self.numTimePred):
            # PYTORCH 43
            self.forError[:,:,ahead] = self.forMean[:,:,ahead] -  self.data.ts[self.outSampleEmb_index]
            # Original
            # self.forError[:,:,ahead] = self.forMean[:,:,ahead] -  self.data.ts[self.outSampleEmb_index]

        self.forErrorComputed = True
        #(self.forMean[:,:,0]-self.inSampleY_mean.cpu())/self.inSampleY_std.cpu()

    def compute_MSPE(self):
        if(not self.forErrorComputed):
            self.compute_forecast_error()

        # PYTORCH 37
        MSPE = torch.nanmean(self.forError**2,dim =(0,1))
        #forcast = self.forMat.numpy()
        #actual = self.data.ts[self.outSampleEmb_index].numpy()
        return MSPE
        # Original
        # return np.Nanmean(self.forError**2,axis = (0,1))
    def persistence_compute(self):
      per_err = torch.zeros_like(self.forMean)
      for i in range(self.numTimePred):
        per_err[:,:,i] = self.data.ts[self.outSampleEmb_index] - self.data.ts[self.outSampleEmb_index - i - 1]
        per_err[torch.isnan(self.forMean)] = torch.nan

        per_mse = torch.nanmean(per_err**2,dim=(0,1))
      print(per_mse) 
      return per_mse
