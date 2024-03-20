delta=seq(0,1,0.01)
load("cov_sample.rda")
load("covar_all.rda")
print("shrinkage for lead one")
load("loc_all.rda")
loc_all = cbind(loc_all,1:53333)
for(i in seq(0,0.5,0.1)){
  if (i == 0){
  load("m4p.rda")
  print("start!")
  h=loc_all[which(loc_all[,1]> 44.98 & loc_all[,1]< 45.1),3]
  v=loc_all[which(loc_all[,2]> 19.9 & loc_all[,2]< 20),3]
  index = intersect(h,v)
  print(length(index))
  }
 else{
	load(paste0("m",i,".rda"))
	print("start!")
	h=loc_all[which(loc_all[,1]> (45-i) & loc_all[,1]<(45+i)),3]
	v=loc_all[which(loc_all[,2]> (20-i) & loc_all[,2]< (20+i)),3]
	index = intersect(h,v)
	print(length(index))
 }

  t_m = mean(m)
  m =sample(m, 1000, replace = F)
  for(k in delta){
    print(paste0("The current shrinkage is : ",k))
    cov_spa = k*covar_all + (1 - k)* cov_sample

    cov_spa = sum(cov_spa[index,index])/(length(index)^2)

    lower = t_m - 1.96 * sqrt(cov_spa)
    upper = t_m + 1.96 * sqrt(cov_spa)

   print(lower)
   print(upper)

  count = 0
  for (j in m){

      if ( (j > lower) & (j < upper)){

      count = count + 1

     }

    }
print(count/1000)

     }
print("====================================================================")

}
