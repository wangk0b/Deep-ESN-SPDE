library(ggplot2)
##################################################

x = data.frame(MSPE = c(B.ESN[,1],ESN[,1],VAR[,1],LSTM[,1],GRU[,1],PER[,1]), Method = rep(c("B-ESN","ESN","VAR","LSTM","GRU","PER"),each=100))

p1 = ggplot(x, aes(x=Method, y=MSPE,color = Method)) + geom_boxplot(outlier.colour="red",notch = T,size = 1)+ggtitle("One-hour Lead")+
	theme(legend.position = "none",text = element_text(size=30),plot.title = element_text(hjust = 0.5))+xlab("")+ylim(c(1.75,3.3))

x = data.frame(MSPE = c(B.ESN[,2],ESN[,2],VAR[,2],LSTM[,2],GRU[,2],PER[,2]), Method = rep(c("B-ESN","ESN","VAR","LSTM","GRU","PER"),each=100))

p2 = ggplot(x, aes(x=Method, y=MSPE,color = Method)) + geom_boxplot(outlier.colour="red",notch = T,size = 1)+ggtitle("Two-hour Lead")+
	theme(legend.position = "none",text = element_text(size=30),plot.title = element_text(hjust = 0.5))+xlab("")+ylab("")+ylim(c(1.75,3.3))

x = data.frame(MSPE = c(B.ESN[,3],ESN[,3],VAR[,3],LSTM[,3],GRU[,3],PER[,3]), Method = rep(c("B-ESN","ESN","VAR","LSTM","GRU","PER"),each=100))

p3 = ggplot(x, aes(x=Method, y=MSPE,color = Method)) + geom_boxplot(outlier.colour="red",notch = T,size = 1)+ggtitle("Three-hour Lead")+
	      theme(legend.position = "none",text = element_text(size=30),plot.title = element_text(hjust = 0.5))+xlab("")+ylab("")+ylim(c(1.75,3.3))

      plot_grid(p1,p2,p3, nrow = 1)


      #lorenz 96

x = data.frame(MSPE = c(B.ESN[,1], ESN[,1], VAR[,1],LSTM[,1],GRU[,1],PER[,1]), Method = rep(c("B-ESN","ESN","VAR","LSTM","GRU","PER"),each=100))

p1 = ggplot(x, aes(x=Method, y=MSPE,color = Method)) + geom_boxplot(outlier.colour="red",notch = T,size = 1)+ggtitle("One-hour Lead")+
	theme(legend.position = "none",text = element_text(size=30),plot.title = element_text(hjust = 0.5))+xlab("")+ylim(c(0,7.5))

x = data.frame(MSPE = c(B.ESN[,2], ESN[,2],VAR[,2],LSTM[,2],GRU[,2],PER[,2]), Method = rep(c("B-ESN","ESN","VAR","LSTM","GRU","PER"),each=100))

p2 = ggplot(x, aes(x=Method, y=MSPE,color = Method)) + geom_boxplot(outlier.colour="red",notch = T,size = 1)+ggtitle("Two-hour Lead")+
	theme(legend.position = "none",text = element_text(size=30),plot.title = element_text(hjust = 0.5))+xlab("")+ylab("")+ylim(c(0,7.5))

x = data.frame(MSPE = c(B.ESN[,3], ESN[,3],VAR[,3],LSTM[,3],GRU[,3],PER[,3]), Method = rep(c("B-ESN","ESN","VAR","LSTM","GRU","PER"),each=100))

p3 = ggplot(x, aes(x=Method, y=MSPE,color = Method)) + geom_boxplot(outlier.colour="red",notch = T,size = 1)+ggtitle("Three-hour Lead")+
	theme(legend.position = "none",text = element_text(size=30),plot.title = element_text(hjust = 0.5))+xlab("")+ylab("")+ylim(c(0,7.5))

plot_grid(p1,p2,p3, nrow = 1)


#########GRU wind
er = c(1.5831938437545,5.34762255903767,5.2484424223651045,4.336048716306022,8.682889579011789,7.421709342598447,8.889166545603898,1.2067912019940334,0.9463513482843589,0.9045387359214176,0.9299534059503297,0.9457358762800779,1.0036847757052871,1.0131933055572837)
Epoch = rep(c(1,5,10,20,30,40,50),2)
Diagnostics = rep(c("Testing Error","Training Error"),each = 7)

data = data.frame(Epoch = Epoch, Error = er, group = Diagnostics)


# Plot
ggplot(data, aes(x=Epoch, y=Error,group = Diagnostics ,color =Diagnostics)) + geom_line(size = 2)+
	geom_point(size =5) +xlab("Epoch")+
	ylab("MSPE")+theme(text = element_text(size = 20),legend.position = c(0.2,0.8))
###########################LSTM
er = c(1.402461656680709,1.654109794495521,1.8014360463214976,6.203874223117123,8.569799188991036,33.10879247465399,23.67260627145561,0.7731341852790312,0.6748397660775292,0.6580297744304986,0.6300395540500405,0.5930244635796682,0.5692446878008875,0.5456311835516959)
Epoch = rep(c(1,5,10,20,30,40,50),2)
Diagnostics = rep(c("Testing Error","Training Error"),each = 7)

data = data.frame(Epoch = Epoch, Error = er, group = Diagnostics)


# Plot
ggplot(data, aes(x=Epoch, y=Error,group = Diagnostics ,color =Diagnostics)) + geom_line(size = 2)+
	geom_point(size =5) +xlab("Epoch")+
	ylab("MSPE")+theme(text = element_text(size = 20),legend.position = c(0.2,0.8))
#######################################################

##############################################################
#########GRU Lorenz96 9 6
er = c(2.98240388218482,2.678600936678786,2.38685305813449,2.3625365825154665,2.334072823500513,2.335568051841391,2.3909339635216407,2.3247948202056508,2.7428147667017413,2.4662523137864163,2.7335544992331875,2.572863490737262,3.3248184244981847,2.443547881589515,2.0875854514567216,1.5813720464258827,1.3344430212532923,1.199150975042947,1.1620500555474111,1.094480153392343,1.0336914096815566,0.9895965583034988,0.9477036644579919,0.8974254606140123)
Epoch = rep(c(1,5,10,20,30,40,50,60,80,100,150,200),2)
Diagnostics = rep(c("Testing Error","Training Error"),each = 12)

data = data.frame(Epoch = Epoch, Error = er, group = Diagnostics)


# Plot
ggplot(data, aes(x=Epoch, y=Error,group = Diagnostics ,color =Diagnostics)) + geom_line(size = 2)+
			      geom_point(size =5) +xlab("Epoch")+
			        ylab("MSPE")+theme(text = element_text(size = 15),legend.position = c(0.2,0.8))+  geom_vline(xintercept=60, linetype="dashed", 
															                                                                                                    color = "red", size=2)
#########################################################LSTM 96 
er = c(4.105831750074224,3.08798615537097,2.74093531963214,2.8647234607581535,2.616066582238855,2.6378925152160413,2.74847025128465,3.051892485450859,3.096515552482412,3.297005694715222,3.401131344800019,3.4221440080422263,3.600651755947643,2.511177792567037,2.33738086325057,1.7637865670929862,1.3335731552449872,1.1001143460727305,0.9529414162021107,0.8494949712696601,0.7741850959195362,0.7136715208932067,0.6457345921718731,0.563887967708263)
Epoch = rep(c(1,5,10,20,30,40,50,60,80,100,150,200),2)
Diagnostics = rep(c("Testing Error","Training Error"),each = 12)

data = data.frame(Epoch = Epoch, Error = er, group = Diagnostics)


# Plot
ggplot(data, aes(x=Epoch, y=Error,group = Diagnostics ,color =Diagnostics)) + geom_line(size = 2)+
	geom_point(size =5) +xlab("Epoch")+
	ylab("MSPE")+theme(text = element_text(size = 20),legend.position = c(0.2,0.8))+  geom_vline(xintercept=30, linetype="dashed", 
												     color = "red", size=2)
################################# Batch
l_one=c(0.8531274,0.83817095,0.8496156,0.8382554,0.8425437,0.83557665,0.8272839,0.7745322)
l_two = c(1.2373132,1.2139974,1.1347847,1.1174269,1.0952263,1.0661513,1.0541636,1.0475489)
l_three =c(1.4815171,1.4544271,1.3535889,1.3323969,1.3060216,1.2725313,1.2580599,1.2322781)

data = data.frame(MSPE = c(l_one,l_two,l_three), Lead = rep(c("One-hour","Two-hour","Three-hour"),each=8),Batch =rep(c(200,150,100,80,60,30,10,1),3))
ggplot(data, aes(x=Batch, y=MSPE,group = Lead ,color =Lead)) + geom_line(size = 2)+
	geom_point(size =5)+scale_x_reverse()+theme(text = element_text(size = 10),legend.position = c(0.2,0.8))

#########################graph ESN + SPDE complexity analysis 
cuda_time = c(511.7917938642204, 541.7507910821587, 590.2080266159028,645.797610450536,693.3409192021936,1162.8756216801703)/3600
cpu_time = c(352.87387852184474,603.5355568770319,1076.123510396108,1313.1926531381905,1728.0408172942698,3369.0459273587912)/3600
nodes = c(100,500,1500,2500,3500,5000)

x = data.frame(nodes = nodes, time = c(cuda_time,cpu_time), Hardware = rep(c("GPU","CPU"),c(6,6)))

ggplot(data=x, aes(x=nodes, y=time, group=Hardware, color =Hardware)) +
	geom_line(size = 2)+
	geom_point(size =5) + ylim(c(0,4000/3600))+xlab(expression(n[h]))+
	ylab("time(hr)")+theme(text = element_text(size = 25),legend.position = c(0.2,0.8))+labs(color=NULL)

#==============================================================

cuda_time = c(535.1397367212921,535.0358299519867,626.4092476330698,652.1049454063177,707.3155843969434,1178.811598578468 )
cpu_time = c(341.6388343926519,653.5523922014982,1307.0613240990788,1820.716846941039, 2180.2371745761484, 3340.0953559353948)
nodes = c(100,500,1500,2500,3500,5000)

x = data.frame(nodes = nodes, time = c(cuda_time,cpu_time), Hardware = rep(c("cuda","cpu"),c(6,6)))

ggplot(data=x, aes(x=nodes, y=time, group=Hardware, color =Hardware)) +
	geom_line(size = 2)+
	geom_point(size =5) +xlab(expression(n[h]))+
	ylab("time(s)")+theme(text = element_text(size = 20),legend.position = c(0.2,0.8))+scale_y_continuous(limits = c(0,4000),position = "right", breaks = c(0,1000,2000,3000,4000))
#=============================================================================================
cuda_time = c(570.4527343846858,579.4262953717262,678.1512677893043, 695.5537666231394,768.2579432073981,1241.421867787838 )
cpu_time = c(366.0160930734128,682.391818953678,1139.6064818315208,1394.045615227893, 1892.8408322688192, 3541.342435080558)
nodes = c(100,500,1500,2500,3500,5000)

x = data.frame(nodes = nodes, time = c(cuda_time,cpu_time), Hardware = rep(c("cuda","cpu"),c(6,6)))

ggplot(data=x, aes(x=nodes, y=time, group=Hardware, color =Hardware)) +
	geom_line(size = 2)+
	geom_point(size =5) + ylim(c(0,4000))+xlab(expression(n[h]))+
	ylab("")+theme(text = element_text(size = 20),legend.position = c(0.2,0.8))
#=============================================================================================
cuda_time = c(617.8573581874371, 624.4783828277141,728.5914238672704,769.9324534870684, 839.3508222606033,1333.1908829323947)/3600
cpu_time = c(418.055852400139,706.8826102185994,1259.1474501993507,1655.1980903167278,2002.5512909255922,3613.5118960365653)/3600
nodes = c(100,500,1500,2500,3500,5000)

x = data.frame(nodes = nodes, time = c(cuda_time,cpu_time), Hardware = rep(c("GPU","CPU"),c(6,6)))

ggplot(data=x, aes(x=nodes, y=time, group=Hardware, color =Hardware)) +
	geom_line(size = 2)+
	geom_point(size =5) +xlab(expression(n[h]))+
	ylab("")+theme(text = element_text(size = 25),legend.position = c(0.2,0.8)) + scale_y_continuous(limits = c(0,4000/3600))+labs(color=NULL)
#===============================================================================================================
require(scales)
scaleFUN <- function(x) sprintf("%.1f", x)
point <- format_format(big.mark = " ", decimal.mark = ".", scientific = FALSE)
cuda_time = c(770.5898961368948,809.5429883189499,893.3321500048041,1111.2242320664227,1333.6503035537899,2068.8128586467355,11735.964401995763,63693.25545997545)/3600
cpu_time = c(1923.103407740593,1680.5690249148756,1896.2520030941814,2327.7701089102775,3245.0827522668988,4871.581576690078,21111.924678958952,88489.5353303533)/3600
nodes = c(8760,4380,2190,1095,730,365,73,25)

x = data.frame(nodes = nodes, time = c(cuda_time,cpu_time), Hardware = rep(c("GPU","CPU"),c(8,8)))

ggplot(data=x, aes(x=nodes, y=time, group=Hardware, color =Hardware)) +
	geom_line(size = 2)+
	geom_point(size =5) +xlab("b")+
	ylab("")+theme(text = element_text(size = 25),legend.position = c(0.1,0.8))+scale_x_reverse()+scale_y_continuous(trans='log10',labels = comma)+theme(legend.position = c(0.2,0.8))+labs(color=NULL)
#================================================================================================================

