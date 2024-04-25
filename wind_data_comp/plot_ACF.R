library(reticulate)
np = import("numpy")
x = np$load("ESN.npy")

index_sup=read.table('../../locselectR.txt',sep=',')[,1]

library(ncdf4)
nc_data = nc_open('../../wind_residual_all_locations.nc')
wind_all=ncvar_get(nc_data,"wind_residual_all_locations")[26281:35030,index_sup]
nc_close(nc_data)
#acf_l2_real = rep(0,3173) 
acf_r = c()

for(i in 1){
 error = wind_all[i:8750,] - x[i:8750,] 
 for(j in c(13,2500)){
   acf_1 = acf(wind_all[,j],lag.max = 50)$acf
   #acf_l2_real[j] = sqrt(sum((acf_1$acf[2:51])^2))
   acf_2 = acf(error[,j],lag.max = 50)$acf 
   #acf_l2[i,j] = sqrt(sum((acf_2$acf[2:51])^2))
   acf_r = rbind(acf_r,acf_1)
   acf_r = rbind(acf_r,acf_2)
  }

}

#np$save("acf_real.npy",acf_l2_real)
#np$save("acf_predict.npy",acf_l2)
save(acf_r,file = "acf_r.rda")






