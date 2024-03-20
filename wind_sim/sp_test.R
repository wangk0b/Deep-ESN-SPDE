options(warn = -1)
seed_option = commandArgs(trailingOnly = TRUE)
seed_option = strtoi(seed_option)
set.seed(seed_option)
#library(support)
library(INLA)
inla.setOption(inla.mode="experimental")
#inla.setOption(pardiso.license = "DD06512683D38AE3653297F6CD7149CCB792CC971DE83F435D951303",smtp="pardiso")
#library(foreach)
#library(doParallel)
#inla.setOption("paradiso.license", "/my/path/to/pardiso.lic")
#inla.setOption(inla.mode="experimental")
#inla.setOption(pardiso.license = "DD06512683D38AE3653297F6CD7149CCB792CC971DE83F435D951303",smtp="pardiso")
#inla.pardiso()
#inla.pardiso.check()
library(reticulate)


np=import("numpy")
y_predict = np$load('space_data.npy')
loc = np$load('loc_all.npy')
loc_t = np$load("loc_te.npy")
ind_te = loc_t[,1]
loc_t = loc_t[,2:3]
#t_l = dim(y_predict)[1]
#s_l dim(y_predict)[2]

#real data
#wind_time = np$load("time_data.npy")
#wind_space = np$load("space_data.npy")

#wind = wind_time + wind_space
#remove(wind_time)
#remove(wind_space)

#loc = np$load("loc_all.npy")

if(seed_option == 1){

 loc_p = np$load("loc_sp.npy")
 index_p = loc_p[,1]
 loc_p = loc_p[,2:3]
}else if(seed_option == 2){
  loc_p = np$load("loc_grid.npy")
  index_p = loc_p[,1]
  loc_p = loc_p[,2:3]

}else{
  loc_p = np$load("loc_rand.npy")
  index_p = loc_p[,1]
  loc_p = loc_p[,2:3]
}

#index=setdiff(1:10000,index_p)
#loc_t = loc[index,]


mesh=inla.mesh.2d(loc.domain = loc, max.edge = c(0.01,1), cutoff = 0.08)
print(mesh$n)
p=mesh$loc[,1]*mesh$loc[,2]
if(seed_option == 1){
nu=0.5
}else{
nu = 0.5
}
alpha=nu+1
logkappa0=log(8*nu)/2
logtau0=(lgamma(nu) - lgamma(alpha) -  log(4 * pi)) / 2
logtau0 = logtau0 - nu*logkappa0
spde = inla.spde2.matern(mesh,
			 B.tau = cbind(logtau0, -sin(p),-cos(p),nu*sin(p),nu*cos(p)),
			 B.kappa = cbind(logkappa0, 0, 0,-sin(p),-cos(p)),
			 theta.prior.mean = rep(0, 4),
			 theta.prior.prec = rep(1, 4))

A=inla.spde.make.A(mesh,loc_p)
Apred =  inla.spde.make.A(mesh,loc_t)
#MSPE = c()

 #cl = makeCluster(30)
 #registerDoParallel(cl)
library(INLA)
inla.setOption(inla.mode="experimental")
sample= y_predict[index_p]
	stack1 = inla.stack(
	data = list(y = sample),
	    A = list(A, 1),
	    effects = list(
	    i=1:spde$n.spde,
	    beta0=rep(1,length(sample))),
	    tag = 'knots')
  #start=Sys.time()
  res5=inla( y ~ 0 + beta0 + f(i,model=spde),
	    data = inla.stack.data(stack1),
	    control.predictor=list(A=inla.stack.A(stack1)))


nprec=dim(loc_t)[1]
stackp = inla.stack(
		    data = list(y = rep(NA, nprec)),
		    A = list(Apred, 1),
		    effects = list(
			i=1:spde$n.spde,
			beta0=rep(1,nprec)),
		    tag = 'pred')

stack2 <- inla.stack(stack1, stackp)

res_p=inla( y ~ 0 + beta0 + f(i,model=spde),
	             data = inla.stack.data(stack2),
		     control.mode=list(theta=res5$mode$theta, restart = FALSE),
		     control.predictor=list(
		    A=inla.stack.A(stack2),compute=F))

     pred_index=inla.stack.index(stack2,"pred")$data
#predict_values_all[iter,]=res_p$summary.fitted.values[pred_index,1]
     p=res_p$summary.fitted.values[pred_index,1]
     

 
 #test_data = matrix(NA, nrow = t_l, ncol = 3600)
 #test_data[,index_p] = y_predict[,,j]
 #test_data[j:t_l,index] = predict 
 MSPE = mean((y_predict[ind_te] - p)^2)

if (seed_option ==  1){
 write(MSPE, file = "sp_mspe.txt", append = TRUE)
}else if(seed_option == 2){
 write(MSPE, file = "grid_mspe.txt", append = TRUE)

}else{
 write(MSPE, file = "rand_mspe.txt", append = TRUE)

}



#x = runif(100,0,1)
#y = runif(100,0,1)
#loc = expand.grid(x,y)

#p = sp(1600,2,dist.samp = loc)






