#load("collapes_ensem.rda")
#compute sample covariance matrix 
#sample_cov = 0
#dim(r)
#index=sample(1:87500,100,replace= F)
#for(i in index){
# sample_cov = sample_cov + outer(r[i,],r[i,])
# if(i %% 10 == 0){
# print("success!")
# }
#}

#sample_cov = sample_cov/100
#save(sample_cov,file="cov_sample.rda")

#m_vec = c()

#for(i in 1:100){
#load(paste0("SPDE_predict_leadone",i,".rda"))
#m_vec = c(m_vec,mean(hold_all))
#rm(hold_all)
#print(paste0("process: ",i))
#}
#save(m_vec,file="leaone_means.rda")

#grid search for the shrinkage parameter
#load("covar_all.rda")
#load("cov_sample.rda")
#load("overall_colm10_leadone.rda")
delta=seq(0,1,0.01)
#t_m = mean(m)
#m = sample(m,1000,replace=F)
#r = seq(0.01,1,0.1)
#var = matrix(0,53333,53333)
#var_samp = matrix(0,53333,53333)
#diag(var_samp)=sqrt(diag(cov_sample))
#diag(var)=sqrt(diag(covar_all))
#covar_all = solve(var)%*%covar_all%*%solve(var)
#cov_sample = solve(var_samp)%*%cov_sample%*%solve(var_samp)
#perform graphical lasso
#library(glasso)
#for(k in r){
#  print(paste0("The current regularization term is: ",k))
#  covar_all = glasso(cov_sample,k)$w
#size = c(50,100,500,1000,3000,10000,30000)

#for(k in size){

#print(paste0("The current size is : ",k))

#compute mean
library(reticulate)
np=import("numpy")
ensembles=np$load('ESN_ensembles.npy')[,3:8750,,3]
loc_select = read.table("locselectR.txt",sep=",")[,1]
loc_pred = setdiff(1:53333,loc_select)
print(length(loc_pred))
load("loc_all.rda")
loc_all =cbind(loc_all,1:53333)
#for(i in seq(0.1,0.5,0.1)){
 h=loc_all[which(loc_all[,1]> 44.98 & loc_all[,1]< 45.1),3]
 v=loc_all[which(loc_all[,2]> 19.9 & loc_all[,2]< 20),3]
 index = intersect(h,v)
 m=c()
for (j in 1:100){
  hold = matrix(NA,8758,53333)
  hold[,loc_select] = ensembles[j,,]
  load(paste0("SPDE_predict_leadthree",j,".rda"))
  hold[,loc_pred] = predict_values_all
  hold = hold[,index]
  m = c(m,rowMeans(hold))
}
save(m,file="mthree4p.rda")
#}
 
#print(paste0("finished mean for",length(index)))

# t_m = mean(m)
# m = sample(m,1000,replace=F)
# for(k in delta){
#    print(paste0("The current shrinkage is : ",k))
#    cov_spa = k*covar_all + (1 - k)* cov_sample2

#cov_spa = var_samp%*%cov_spa%*%var_samp

#    cov_spa = sum(cov_spa[index,index])/(length(index)^2)

 #   lower = t_m - 1.96 * sqrt(cov_spa)
 #   upper = t_m + 1.96 * sqrt(cov_spa)

 #  print(lower)
 #  print(upper)

 # count = 0
 # for (j in m){
 
 #     if ( (j > lower) & (j < upper)){
 
 #     count = count + 1
 
 #    }

 #   }
#print(count/1000)

 #    }
#} 
  




