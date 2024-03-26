mport numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
sp = np.loadtxt("sp_mspe.txt")
grid = np.loadtxt("grid_mspe.txt")
rand = np.loadtxt("rand_mspe.txt")
reference = np.mean(sp)
d2 = {'SP':sp,'Grid':grid,'Rand':rand}
plt.figure(figsize = (10,8))
ax = plt.subplot(111)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.boxplot(d2.values())
#plt.axhline(reference,c = 'r')
ax.set_xticklabels(d2.keys())
#plt.show()
plt.savefig("r_select.pdf")



loc = np.load("loc_all.npy")
v = np.load("space_data.npy")
plt.figure(figsize = (10,8))
ax = plt.subplot(111)
ax.scatter(loc[:,0],loc[:,1], c = v,marker='.',s=10)


l_one=np.array([0.8506041,0.8375816,0.8521318,0.8420999,0.836941,0.8292243,0.8154739,0.77345514])
l_two = np.array([1.235371,1.2161899,1.1364207,1.0829782,1.0644407,1.0576143,1.0515991,1.0470042])
l_three =np.array([1.4802411,1.4589844,1.3537959,1.2897661,1.2676413,1.2620662,1.2596314,1.2757255])
avg = (l_one+l_two+l_three)/3
b = np.array([200,150,100,50,25,10,5,1])
time = np.array([13.609,13.394,13.833,13.749,15.139,17.266,20.192,45.188])
print(avg)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(b, avg, marker = 'o',c = 'blue',markersize=2)
ax2.plot(b, time, marker = 'v',c = 'Red',markersize=2)

ax1.set_xlabel('b')
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.xaxis.grid(color='gray', linestyle='dashed')
ax1.set_ylabel('MSPE',c = "blue")
ax2.set_ylabel('time(s)', c = 'Red')
ax1.set_xlim(200, 0)
plt.savefig("complex.pdf")
#plt.show()



quantile = np.arange(0.025,1,0.025)
df = np.load('power_quantiles.npz')
esn_power_turbine_height_quantile = df['esn_power_turbine_height_quantile']
fig = plt.figure(figsize = (10,8))
ax = plt.subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.xaxis.grid(color='gray', linestyle='dashed')
plt.plot(quantile, esn_power_turbine_height_quantile, marker = 'v', linewidth = 2,c = "Red",mfc='blue',markersize=15)
plt.xticks([0.025, *np.arange(0.1,1,0.1), 0.975], [2.5, *np.arange(10,100,10), 97.5], fontsize = 20)
plt.yticks(fontsize = 15)
tx = ax.yaxis.get_offset_text()
tx.set_fontsize(20)
plt.ylabel('Annual Sum of Absolute Differences\nin Wind Energy (kWh)', fontsize = 25)
plt.xlabel('Quantiles (%)', fontsize = 25)
plt.tight_layout()
plt.savefig("power.pdf")



import numpy as np

x = np.load("quantiles_all.npz")
print(x['forErrorQuantile_all'].shape)
import matplotlib.pyplot as plt
import matplotlib
ESN = np.load("VAR_wind_energy.npy")
VAR = np.load("esn_wind_energy.npy")
LSTM = np.load("LSTM_wind_energy.npy")
GRU = np.load("GRU_wind_energy.npy")
LAT = np.load("lat_wind_energy.npy")
data = np.zeros((75,5))

data[:,0] = np.nanmean(ESN,axis = 1)
data[:,1] = np.nanmean(VAR,axis = 1)
data[:,2] = np.nanmean(LSTM,axis = 1)
data[:,3] = np.nanmean(GRU,axis = 1)
data[:,4] = np.nanmean(LAT,axis = 1)



n_bins = 10
matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize = (10,8))
colors = ['red', 'tan', 'lime','blue','violet']
l = ['SPDE-ESN','SPDE-VAR','SPDE-LSTM','SPDE-GRU','LAT-ESN']
plt.hist(data, n_bins, density=False, histtype='bar', color=colors, label=l)
plt.legend(prop={'size': 18})
plt.xlabel('Averaged absolute Differences in wind energy (KWh)')
#plt.show()
plt.savefig("hist.pdf")











from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from matplotlib import cm
#import pyreadr
#x = pyreadr.read_r("mean_sd_wind.rda")
#x = x["all"].to_numpy()
#x = x[0:53333,:]
#x = x[0:53333,:]
#x = np.load("esn_mse_all_locations.npz")
#x = x['esn_mse_all_locations']
#y = np.load("MSPE_three.npy")
#relative = (x - y)/x
#print(relative.mean(axis=0))
#load files
#wind_mean_all=np.loadtxt('mean_loc_two.txt',delimiter=',')
import pandas as pd
def encode(df,columns):
        return  df.get_dummies(df,columns = columns)
        data = {"color":["red","blue"]}
        df = pd.DataFrame(data)
        en = encode(df,["color"])
        print(en)

        index = np.loadtxt("locselectR.txt", delimiter=",",dtype= np.int32,usecols=(0)) - 1
        wind_locations = np.loadtxt('wind_locations_all.txt',delimiter=' ',usecols=(0,1))
        #wind_locations = np.loadtxt('locselectR.txt',delimiter=',',usecols=(1,2))
        #wind_locations = wind_locations[index,:]
        #index = np.arange(0,53333)
        #index = np.random.choice(index,5000,replace=False)
        meridians = np.arange(35,59,5);
        parallels = np.arange(17,33,5);
        m = Basemap(resolution='l',llcrnrlon=33.6, llcrnrlat=15.4,
            urcrnrlon=56.6,urcrnrlat=33.2);
            #plt average
            plt.figure(figsize = (10,8))
            ax = plt.subplot(111)
            imc=m.scatter(wind_locations[index,0],wind_locations[index,1],marker='.',s=10,color="r")
            #imc=m.scatter(wind_locations[:,0],wind_locations[:,1],marker='.',c=relative[:,2],cmap=cm.RdBu_r, s=1)
            #m.scatter(np.array([35.30612,38.55765,43.83173,51.08688]),np.array([28.37077,30.13631,23.10381,20.60040]),marker='x', s=100,color="k")
            m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0, fontsize = 20)
            m.drawparallels(parallels,labels=[True,False,False,True], linewidth=0, fontsize = 20)
            m.drawcoastlines(linewidth=1, color="black")
            m.drawcountries(linewidth=1, color="black")
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #cbar = plt.colorbar(imc, cax=cax)
            #cbar.ax.tick_params(labelsize = 20)
            ax.set_title('',fontsize=30)
            ax.set_aspect('auto')
            #plt.clim(-0.8,0.8)
            #plt.show()
            #plt.savefig('knots.pdf')
            #exit()

            #plt knots
            #plt.figure(figsize = (10,8));
            #ax=plt.subplot(111);
            #m.scatter(knot_lcations[:,0],knot_lcations[:,1],marker='.',color='red',s=10);
            #m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=0, fontsize = 20);
            #m.drawparallels(parallels,labels=[True,False,False,True], linewidth=0, fontsize = 20);
            #m.drawcoastlines(linewidth=1, color="black");
            #m.drawcountries(linewidth=1, color="black");
            #ax.set_title('Selected Knots for DESN',fontsize=30);
            #ax.set_aspect('auto');
            #plt.show()
            #plt.savefig('knots.pdf')
        


