library(reticulate)
library(LatticeKrig)
library(foreach)
library(doParallel)

np = import("numpy")
ensembles=np$load('ESN_ensembles.npy')
y_predict = apply(ensembles,c(2,3,4),mean)
print(dim(y_predict))
remove(ensembles)

loc_s = read.table("locselectR.txt",sep = ",")
index_s = loc_s[,1]

library(ncdf4)
nc_data = nc_open('wind_residual_all_locations.nc')
lat_all=ncvar_get(nc_data,"lat_all_locations")
lon_all = ncvar_get(nc_data,"lon_all_locations")
nc_close(nc_data)

loc_all = cbind(lon_all,lat_all)
index_predict = setdiff(1:53333,index_s)
x_p = loc_all[index_predict,]
x = loc_all[index_s,]
L = LKrigSetup(loc_all,NC = 22, nlevel = 2, a.wght = 4.5, nu = 0.5)
print(L)
cl = makeCluster(30)
registerDoParallel(cl)

for(i in 1:3){
 y = y_predict[(i:8760),,i]

result = foreach(iter = i:8750, .combine = rbind) %dopar%{
 library(LatticeKrig)
 fit = LatticeKrig(x, y[iter,], LKinfo = L)
 p = predict(fit,x_p)
 p
}
np$save(paste0("LatKrig_Pred",i,".npy"),result)
}

stopCluster(cl)
print("Done!!")
