# Deep Learning Project CS-E4890
# Extended-Kalman-Filter for Neural Networks

This project is to explore the Extended Kalman Filter (EKF) method for parameter/weight update in a neural network and compare the results with traditional gradient descent methods.

The results are compared using 3 datasets
  - Abalone
  - Bike Sharing
  - Wine Quality
These are datasets found in the UCI machine learning repository.

We have referred papers such as [[1]](https://www.sciencedirect.com/science/article/pii/S0925231219300980?via%3Dihub) and [[2]](https://link.springer.com/article/10.3103/S1060992X14020088) 

We utilize 2 jupyter notebooks that showcase EKF and SGB backpropagation. The gradient descent notebook utilizs pytorch to create and train the network on the same data set as that used in EKF. The EKF notebook creates a netowork using the custom `knn.py` class and trains it utilizing only `numpy` matrix calculations. The datasets are also provided in the `data` folder and a couple of saved _EKF_ models are also present to test in the `saved_models` folder. 


Based on https://github.com/jnez71/kalmaNN 
