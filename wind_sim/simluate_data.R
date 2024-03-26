#seed_option = commandArgs(trailingOnly = TRUE)
#seed_option = strtoi(seed_option)

#set.seed(seed_option)
library(reticulate)
#library(arfima)
library(support)
library(nabor)
#library(foreach)
#library(doParallel)
#define the matern correlation function
matern = function(sigma,beta,nu,h){
 value = sigma/2^(nu - 1)/gamma(nu)*(h/beta)^nu*besselK(h/beta,nu)
 return(value)

}


np = import("numpy")

#print(matern(1,0.01,0.5,0.1))

#loc = read.table("../locselectR.txt",sep=",")

#loc = loc[,2:3]

#simulate concentrated location 

#training
loc = np$load("loc_all.npy") 
#testing 

N = 3200
ind = 1:N
index_b1 = ind[((loc[,1] > 0) & (loc[,1] < 0.5))& ((loc[,2] > 0.5) & (loc[,2] < 1))]
index_b2 = ind[((loc[,1] > 0.5) & (loc[,1] < 1))& ((loc[,2] > 0.5) & (loc[,2] < 1))]
index_b3 = ind[((loc[,1] > 0) & (loc[,1] < 0.5))& ((loc[,2] > 0) & (loc[,2] < 0.5))]
index_b4 = ind[((loc[,1] > 0.5) & (loc[,1] < 1))& ((loc[,2] > 0) & (loc[,2] < 0.5))]
#loc = expand.grid(x,y)
d_mat = dist(loc,diag=T,upper=T)
H_one = as.matrix(matern(1,0.03,0.5,d_mat))
diag(H_one) = 1

H_two = as.matrix(matern(1,0.05,1,d_mat))
diag(H_two) = 1
###################
#sample from the two covariance matrices
#######################################
#t = 1000
#sim_data = matrix(NA,nrow = t, ncol = N)

L_one = t(chol(H_one))
L_two = t(chol(H_two))
#########################################
#N_one = L_one%*%rnorm(N)
#N_two = L_two%*%rnorm(N)

#sim_data[1,] = 0.5*N_one + 0.5*N_two

#phi = matrix(runif(N^2,-0.5,0.5),3173,3173)

#for(i in 2:t){
#  N_one = L_one%*%rnorm(N)
#  N_two = L_two%*%rnorm(N)
# if(t %% 2 == 0)
# {
#  sim_data[i,] = rnorm(1,0.1,0.3)*sim_data[i-1,] + 0.5*N_one + 0.5*N_two
# }
# else{
# sim_data[i,] = rnorm(1,0.1,0.3)*sim_data[i-1,] + rnorm(1,0.1,0.3)*sim_data[i-2,] + 0.5*N_one + 0.5*N_two
# }

#}
#pb = txtProgressBar(min = 1, max = N, initial = 1,style = 3) 

#cl = makeCluster(30)
#registerDoParallel(cl)


#time_data = foreach(i = 1:N, .combine = cbind) %dopar% {
# library(arfima)
# set.seed(seed_option) 
# sim_data = arfima.sim(1000, model = list(phi = c(0.3,-0.1), theta=c(0.1,-0.3), dfrac = .3, dint = 0))
 #setTxtProgressBar(pb,i)
# p = sim_data
# p
#}
#close(pb)
#stopCluster(cl)

#pb.two = txtProgressBar(min = 1, max = t, initial = 1, style = 3)
#set.seed(seed_option)
N_one = L_one%*%rnorm(N)
N_two = L_two%*%rnorm(N)

B1 = 0.7*N_one[index_b1] + 0.3*N_two[index_b1]
B2 = 0.6*N_one[index_b2] + 0.4*N_two[index_b2]
B3 = 0.9*N_one[index_b3] + 0.1*N_two[index_b3]
B4 = 0.55*N_one[index_b4] + 0.45*N_two[index_b4]
sim_data = c(B1,B2,B3,B4)
#setTxtProgressBar(pb.two,j)
##########################################################################
#print(sim_data)
#np$save("time_data.npy",time_data)
np$save("space_data.npy",sim_data)
print("Done!")










