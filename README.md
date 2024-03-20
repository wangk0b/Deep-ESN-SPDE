# Deep-ESN-SPDE
Deep Sparse Recurrent Neural Network + Stochastic Partial Differential Equation for Large-Scale Wind Data
This project is based on the high spatio-temporal resolution Saudi wind speed dataset provided in the [KAUST Library](https://repository.kaust.edu.sa/handle/10754/667127).
To access the data, please follow the instructions detailed in the readme file of [https://github.com/hhuang90/KSA-wind-forecast?tab=readme-ov-file](https://github.com/hhuang90/KSA-wind-forecast?tab=readme-ov-file).
The ESN_Batch.py contains the B-ESN (Batch Echo State Network) model and forecast_ESN.py is the script to call the B-ESN model. Notice that forecast_ESN.py will parse arguments from the system in a Linux environment. 
