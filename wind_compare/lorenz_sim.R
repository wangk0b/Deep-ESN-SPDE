library(reticulate)
np = import("numpy")
#@####Lorenz 96 Functions from paper ###############
####################################################

mod = function(i,n) return((i-1)%%n+1)
##########Standard deterministic model
lorenz96 = function(x0, theta, dt, n, M){
	  ####################################################
	  # Take M steps of the L96 model with step size dt
	  # Uses 1st-order Euler scheme
	  #---------------------------------------------------
	  # x0: (nx1) initial conditions
	  # theta: scalar forcing parameter (often called F)
	  # dt: internal step size
	  # n: number of spatial locations (n=40 is standard)
	  # M: number of internal steps:   M=delta/dt
	  ####################################################
	  ###### This function is just for one time period
	  ###### mod operator makes
  xx = x0
  dx = rep(0,n)
   
    for (j in 1:M){
	        for (i in 1:n){
			      dx[i] = ( (xx[mod(i+1,n)] - xx[mod(i-2,n)]) * xx[mod(i-1,n)] - xx[i] + theta) * dt
         # print(mod(i-1, n))
	    }
      xx = xx + dx
        }
    return(xx)
}

#####################################################
#Creating Lorenz96 dataset using code from paper#####
#####################################################
dataSet="Lorenz96"
if(dataSet =="Lorenz96")
{
	numLocs=81
}

if(dataSet=="Lorenz96")
{

	#####turn datatype on
	dataTypeLorenz=TRUE

	####Set Embedding Indicator
	embedInd=FALSE

	#######set tau
	tau=1 #originally was set as 6 from author's code. Paper mentions how value of 1 produces more accurate results
	   
	numFullDimLocs=numLocs

	truncLength=3001
	tempTrainLen=1800-tau+1

	#############################################
	# Solve with external step size delta=.005
	# (internal step size dt=.005)
	#Larger delta more nonlinear
	#############################################
	deltaLorSim = 0.1

	thetaLorSim = 4.5


	######smaller dt, better approximation?
	dt = .005
	# dt = .01
	TT = 850

	lorenzM = deltaLorSim/dt
	rawNumTimeLor96 = TT/deltaLorSim + 1

	rawDataTemp = matrix(NA,rawNumTimeLor96,numLocs)
	######initial conditions
	rawDataTemp[1,] = rnorm(numLocs)
	######simulate without error

	for (i in 2:rawNumTimeLor96) rawDataTemp[i,] = lorenz96(rawDataTemp[i-1,],thetaLorSim,dt,numLocs,lorenzM)
	strLorenz96=2000
	endLorenz96=4500
	lorenz96ByLen=1
	sigmaEpsTr=.5
	startLoc=1

	rawData=rawDataTemp[strLorenz96:endLorenz96,(startLoc:(startLoc+numLocs-1))]+matrix(rnorm(length(strLorenz96:endLorenz96)*numLocs,0,sigmaEpsTr),nrow=length(strLorenz96:endLorenz96),ncol=numLocs)
}
#print(dim(rawData))
sim.dat = rawData[-1,][1:1000,]
#print(dim(sim.dat))
np$save( 'Lorenz96.npy',sim.dat)
