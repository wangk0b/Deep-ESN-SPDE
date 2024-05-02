# Deep-ESN-SPDE
Deep Sparse Recurrent Neural Network + Stochastic Partial Differential Equation for Large-Scale Saudi Wind Data. <br />
<br />
This project is based on the high spatio-temporal resolution Saudi wind speed dataset provided in the [KAUST Library](https://repository.kaust.edu.sa/handle/10754/667127). <br />
<br />
To access the data, please follow the instructions detailed in the readme file of [https://github.com/hhuang90/KSA-wind-forecast?tab=readme-ov-file](https://github.com/hhuang90/KSA-wind-forecast?tab=readme-ov-file). <br />
<br />
The ESN_Batch.py contains the B-ESN (Batch Echo State Network) model and forecast_ESN.py is the script to call the B-ESN model. Notice that forecast_ESN.py will parse arguments from the system in a Linux environment. <br />
<br />
This repository consists of several directories. <br />
<br />
1. The calibrate directory contains the implementations of the random square selection method and the convex combination of empirical and spatial covariance structure. <br />
2. The wind_compare directory includes the simulation study performed on the Lorenz-96 data. <br />
3. wind_data_comp details the scripts used to benchmark the machine learning and statistical models against our proposed B-ESN + SPDE model on the actual wind data. <br />
4. wind_power is for the wind power assessment and uncertainty quantification. <br />
5. wind_sat_sim and wind_ray_sim are the directories for the simulation study for benchmarking the performance of support points (SP) on both satellite-like and ray-like domains as a spatial reduction method against Grid and Rand. <br />
6. Lastly, Draw contains the codes for drawing all the figures. 

    
