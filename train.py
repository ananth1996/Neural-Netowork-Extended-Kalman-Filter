
#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#%%
import numpy as np
import matplotlib.pyplot as plt
import knn as kalmann
from load import load_bikes_data
from tqdm import tqdm 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from  tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=150

#%% [markdown]
# ##  Load Data

#%%
X,y = load_bikes_data()
print(y.shape)
print(X.shape)

#%% [markdown]
# ## Preprocess Data

#%%
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

#%% [markdown]
# ## Partition Dataset

#%%
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=.5)
X_val,X_test, y_val,y_test = train_test_split(X_test,y_test,test_size=0.5)
print(f'Train Shape: {X_train.shape}')
print(f'Validation Shape: {X_val.shape}')
print(f'Test shape: {X_test.shape}')

#%% [markdown]
# ## Create EKF network 

#%%
n_inputs = X_train.shape[1]

knn_ekf = kalmann.KNN(nu=n_inputs, ny=1, nl=10, neuron='relu')


#%%
nepochs = 100
tolerance = 1e-4
patience = 20
RMS,_=knn_ekf.train(nepochs=nepochs, U=X_train, Y=y_train, 
                    U_val=X_val,Y_val=y_val, method='ekf', 
                    P=100, Q=10e-6, R=10,
                    tolerance=tolerance,patience=patience)


#%%
RMS[-1]


#%%
y_pred= knn_ekf.feedforward(X_test)
error = mean_squared_error(y_test,y_pred)
np.sqrt(error)

#%% [markdown]
# ## Plotting RMS decrease

#%%
plt.plot(range(len(RMS)),RMS,label="Vaidation Error")
plt.xlabel("Epochs")
plt.legend()
plt.yscale('log')

#%% [markdown]
# ## Save file 

#%%
# knn_ekf.save("saved_models/efk_nn_relu_P_100_Q_10e-6_R_10")


