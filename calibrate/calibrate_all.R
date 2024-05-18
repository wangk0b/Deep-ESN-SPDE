load("cov_sample.rda")
load("covar_all.rda")
#library(foreach)
#library(doParallel)

#covar = cov_sample/10

covar_all =  0.64*covar_all + 0.36*cov_sample
#rm(covar)

#cl=makeCluster(40)
#registerDoParallel(cl)


for(j in 1:53333){


v = covar_all[j,j]

loc_one = c()

for (i in 1:100){
 load(paste0("SPDE_predict_leadone",i,".rda"))
 loc_one = c(loc_one,predict_values_all[,j] ) 
 
}

m = mean(loc_one)

#loc_one = sample(loc_one,5000,replace = F)

upper = m + 1.96 * sqrt(v)
lower = m - 1.96 * sqrt(v)

count = 0
for( i in loc_one){

 if( (i < upper) & (i > lower)){
  count = count + 1
 
 }

}
print("++++++++++++++++++++++++++++")
print(count/53333)

}
#stopCluster(cl)
#print(CI)





