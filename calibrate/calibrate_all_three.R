library(ncdf4)
library(reticulate)
#library(foreach)
#library(doParallel)
#library(cobs)
options(warn=-1)
np=import("numpy")
ensembles=np$load('ESN_ensembles.npy')[,,,2]
index = read.table("locselectR.txt",sep=",")[,1]
index_all =setdiff(1:53333,index)

cov_sample2 = 0
#cov_sample3 = 0
#p1 = matrix(NA,8750,53333)
p2 = matrix(NA,8759,53333)
#p3 = matrix(NA,8748,53333)
print("Initilized matrix")
library(ncdf4)
nc_data = nc_open('wind_residual_all_locations.nc')
wind_all=ncvar_get(nc_data,"wind_residual_all_locations")[26281:,]
nc_close(nc_data)
#calibarte the variances
print("start forloop")
for (i in 1:100){
 #compute the residuals for each ensemble

# load(paste0("SPDE_predict_leadone",i,".rda"))
# p1[,index] = ensembles[i,1:8750,,1]
# p1[,index_all] = predict_values_all
# p1 = p1 - wind_all
# m1 = colMeans(p1)
# p1 = sweep(p1,2,m1,FUN="-")
# s1 = apply(p1,2,sd)
# p1 = sweep(p1,2,s1,FUN="/")

 load(paste0("SPDE_predict_leadtwo",i,".rda"))
 p2[,index] = ensembles[i,2:8760,]
 p2[,index_all] = predict_values_all
 rm(predict_values_all)
 p2 = p2 - wind_all[2:8760,]
 m2 = colMeans(p2)
 p2 = sweep(p2,2,m2,FUN="-")
 s2 = apply(p2,2,sd)
 p2 = sweep(p2,2,s2,FUN="/")
 r_index=sample(1:8759,50,replace = F)
 for(k in r_index){
  cov_sample2 = cov_sample2 + outer(p2[k,],p2[k,])
 }

# load(paste0("SPDE_predict_leadthree",i,".rda"))
# p3[,index] = ensembles[i,3:8750,,3]
# p3[,index_all] = predict_values_all
# rm(predict_values_all)
# p3 = p3 - wind_all[3:8750,]
# m3 = colMeans(p3)
# p3 = sweep(p3,2,m3,FUN="-")
# s3 = apply(p3,2,sd)
# p3 = sweep(p3,2,s3,FUN="/")
# r_index=sample(1:8748,50,replace = F)
# for(l in r_index){
# cov_sample3 = cov_sample3 + outer(p3[l,],p3[l,])
# }

# cl = makeCluster(20)
# registerDoParallel(cl)
# s_var = foreach(iter=1:53333,.combine = cbind) %dopar% {
#stopCluster(cl)
print(paste0("One success! ",i))
}
cov_sample2 = cov_sample2 /5000
save(cov_sample2, file = "cov_samp2.rda")
#save(cov_sample3/5000, file = "cov_samp3.rda")
