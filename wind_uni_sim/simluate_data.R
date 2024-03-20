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

#loc = expand.grid(x,y)
d_mat = dist(loc,diag=T,upper=T)
H_one = as.matrix(matern(1,0.03,0.5,d_mat))
diag(H_one) = 1

H_two = as.matrix(matern(1,0.05,1,d_mat))
diag(H_two) = 1
###################
#sample from the two covariance matrices
H_three = as.matrix(matern(1,0.03,1.5,d_mat))
diag(H_three) = 1

#sim_data = matrix(NA,nrow = t, ncol = N)

L_one = t(chol(H_one))
L_two = t(chol(H_two))
L_three = t(chol(H_three))

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


#close(pb)

#pb.two = txtProgressBar(min = 1, max = t, initial = 1, style = 3)

  N_one = L_one%*%rnorm(N)
  N_two = L_two%*%rnorm(N)
  N_three = L_three%*%rnorm(N)
  sim_data = 0.7*N_one + 0.15*N_two + 0.15*N_three
  #setTxtProgressBar(pb.two,j)
##########################################################################
#print(sim_data)
np$save("space_data.npy",sim_data)
print("Done!")










