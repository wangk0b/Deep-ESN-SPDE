library(reticulate)
np = import("numpy")

y = 0

for(i in 1:100){
 load(paste0("../SPDE_predict_leadone",i,".rda"))
 y = y + predict_values_all

}

y=y/100

np$save("ESN_mean_predictone.npy",y)


print("Done!!")
