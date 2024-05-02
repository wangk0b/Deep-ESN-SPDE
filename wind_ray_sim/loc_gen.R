seed_option = commandArgs(trailingOnly = TRUE)
seed_option = strtoi(seed_option)
set.seed(seed_option)

library(support)
library(nabor)
library(reticulate)

np = import("numpy")

Loc1func=function(n){ # ray
  n0=n/16
  loc1=matrix(0,n,2)
  loc1[1:(12*n0),]=matrix(0.25*runif(2*12*n0),12*n0,2)
  loc1[(12*n0+1):(15*n0),]=rbind(cbind(runif(n0,0,0.25),runif(n0,0.25,0.5)),
                                 cbind(runif(n0,0.25,0.5),runif(n0,0.25,0.5)),
                                 cbind(runif(n0,0.25,0.5),runif(n0,0,0.25)))
  loc1[(15*n0+1):(16*n0),]=rbind(cbind(runif(round(n0/3),0,0.5),runif(round(n0/3),0.5,1)),
                                 cbind(runif(round(n0/3)+1,0.5,1),runif(round(n0/3)+1,0.5,1)),
                                 cbind(runif(round(n0/3),0.5,1),runif(round(n0/3),0,0.5)))
  return(loc1)
}

N = 3200
loc=Loc1func(1e4)
index = sample(1:1e4,3200,replace = F)
loc = loc[index,]
#loc = expand.grid(x,y)
#set.seed(1)

#loc = Loc2func(N)
set.seed(0)
test_ind = sample(1:3200,1600,replace = F)
test_ind = sort(test_ind)
train_ind = setdiff(1:3200,test_ind)
train_ind = sort(train_ind)

loc_tr = cbind(train_ind,loc[train_ind,])
loc_te = cbind(test_ind,loc[test_ind,])

loc_sp = sp(100,2,dist.samp = loc_tr[,2:3])$sp
index = knn(loc_tr[,2:3],loc_sp,k=1)$nn.idx
index = train_ind[index]
loc_sp = cbind(index,loc[index,])


index_r = sample(train_ind, 100, replace = F)
loc_r = cbind(index_r,loc[index_r,])

x_g = seq(0,1,length.out = 10)
y_g = seq(0,1,length.out = 10)
loc_g = expand.grid(x_g,y_g)
index_g = knn(loc_tr[,2:3],loc_g,k=1)$nn.idx
index_g = train_ind[index_g]
loc_g = cbind(index_g,loc[index_g,])


np$save("loc_all.npy",loc)
np$save("loc_sp.npy",loc_sp)
np$save("loc_grid.npy",loc_g)
np$save("loc_rand.npy",loc_r)
np$save("loc_tr.npy",loc_tr)
np$save("loc_te.npy",loc_te)

print("Done!!")
