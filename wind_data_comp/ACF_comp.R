library(reticulate)
np = import("numpy")
x = np$load("ESN.npy")

index_sup=read.table('../../locselectR.txt',sep=',')[,1]

library(ncdf4)
nc_data = nc_open('../../wind_residual_all_locations.nc')
wind_all=ncvar_get(nc_data,"wind_residual_all_locations")[26280:35030,index_sup]
nc_close(nc_data)
acf_l2_real = rep(0,3173) 
acf_l2 = matrix(0,3,3173)

for(i in 1:3){
 error = wind_all[i:8750,] - x[i:8750,,i] 
 for(j in 1:3173){
  if(i == 1){
   acf_1 = acf(wind_all[,j],lag.max = 50)
   acf_l2_real[j] = sqrt(sum((acf_1$acf[2:51])^2))
  }
   acf_2 = acf(error[,j],lag.max = 50) 
   acf_l2[i,j] = sqrt(sum((acf_2$acf[2:51])^2))
  }

}

np$save("acf_real.npy",acf_l2_real)
np$save("acf_predict.npy",acf_l2)







