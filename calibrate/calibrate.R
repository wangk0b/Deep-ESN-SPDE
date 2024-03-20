for(num in 10:10){
#location wise
holder=array(NA,c(10,8760,50160))
#perlocation
enum = (1+(num-1)*10):(num*10)
count = 1 
for(i in enum){
  load(paste0('SPDE_predict_leadone',i,'.rda'))
  holder[count,,] = predict_values_all
  count = count + 1
  print("Done:loading Ensembles")
}

#save(holder,file='subsamples.rda')
#q()
#combine the two predictions
#set.seed(1)
index = read.table("locselectR.txt",sep=",")[,1]
index_all =setdiff(1:53333,index)
hold_all = array(NA,dim=c(10,8760,53333))
#load("subsamples.rda")
hold_all[,,index_all]=holder
rm(holder)

library(reticulate)
np=import("numpy")
ensembles=np$load('ESN_ensembles.npy')[enum,1:8760,,1]
hold_all[,,index]= ensembles
rm(index_all)
rm(index)
rm(ensembles)
#load actual values
library(ncdf4)
nc_data = nc_open('wind_residual_all_locations.nc')
wind_all=ncvar_get(nc_data,"wind_residual_all_locations")[26281:35040,]
print("dimensions of real data and predicted data")
dim(wind_all)
dim(hold_all)
#residual
#hold_all = sweep(hold_all,1,wind_all,FUN = "-")
rm(nc_data)
print("Finished assigments and loading real data")
#normalize the ensembles
#step one
for(i in 1:10){
 #compute the residuals
 hold_all[i,,]=hold_all[i,,]-wind_all
 #standardize
 var_ensem = apply(hold_all[i,,],2,sd)
 mean_ensem = apply(hold_all[i,,],2,mean)
 hold_all[i,,]=sweep(hold_all[i,,],2,mean_ensem, FUN = "-")
 hold_all[i,,]=sweep(hold_all[i,,],2,var_ensem, FUN = "/")
 print("Success! standaridization")
}
rm(wind_all)
save(hold_all,file = paste0("STD_ensembles",num,".rda"))
rm(hold_all)
print("iteration done")
}












