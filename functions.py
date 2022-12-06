import numpy as np
import matplotlib.pyplot as plt    
               
## Create and Initialize Network Weights
def initialize_parameters(layer_dims):
    np.random.seed(1)               # to get consistent initial vals 
    num_of_layers = len(layer_dims) # input layer included            
    parameters = {}                 # initialize dictionary 
                
    for i in range(1,num_of_layers):
        parameters['W'+str(i)] = \
            np.random.randn(layer_dims[i],layer_dims[i-1])/ \
        np.sqrt(layer_dims[i-1]) * 0.1
        parameters['b'+str(i)] = np.zeros((layer_dims[i],1))
    return parameters 

## Forward Propagation
def sigmoid(x):           
    y = 1/(1+np.exp(-x))
    return y

def forward_layer(A_prev, W, b, activation_type):
    Z = np.dot(W,A_prev) + b

    if activation_type == 'sigmoid':
        A = sigmoid(Z)
    elif activation_type == 'relu':
        A = np.maximum(0,Z)

    cache = (A_prev,W,Z)          


    return cache, A 
                  
def compute_cost(Y, Y_hat, loss_type='binary_cross_entropy'):
    m = Y.shape[1] 
    
    if loss_type == 'binary_cross_entropy':
        J = (-1./m)*(np.dot(Y,np.log(Y_hat).T) + np.dot(1-Y,np.log(1-Y_hat).T))
    return J


## Backward Propagation
def backward_layer(dA,Z,A_prev,W,activation_type):
    m = A_prev.shape[1] # num of examples is num of columns
    if activation_type == 'relu':            
        dg_dz = np.int64(Z>0)
    elif activation_type == 'sigmoid':
        dg_dz = sigmoid(Z)*(1 - sigmoid(Z))

    dZ = dA * dg_dz
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True) 
                        # keepdims will keep the dimention of the axis we
                        # perform the summation in, instead of eliminating it
    dA_prev = np.dot(W.T,dZ) # dA of the layer to the left 
    
    
    return dA_prev, dW, db

# Optimize: Gradient Descent
def optimize(parameter, grad, learning_rate):
    parameter -= learning_rate*grad

## Prediction        
def predict(parameters, X, Y, hidden_act_type, output_act_type):
    ## Forward Prop
    num_of_layers = len(parameters)//2
    m = X.shape[1]
    # for hidden layers:     
    input_data = X
    for l in range(1,num_of_layers):  
        _, A = forward_layer(input_data, parameters['W'+str(l)], \
                                      parameters['b'+str(l)], hidden_act_type)
        input_data = A 
    # for output layer:     
    _, y_hat = forward_layer(input_data, parameters['W'+str(num_of_layers)], parameters['b'+str(num_of_layers)], output_act_type)
    
    avg = np.sum(y_hat[0]) / y_hat.shape[1]
    predictions = np.int64(y_hat > avg)
    matrix(predictions, Y)

def matrix(X, y):
    print(X)
    X = X[0]
    y = y[0]
    part = X ^ y
    pcount = np.bincount(part)
    tp_list = list(X & y)
    fp_list = list(X & ~y)
    TP = tp_list.count(1)
    FP = fp_list.count(1)
    TN = pcount[0] - TP
    FN = pcount[1] - FP
    evaluation(TP, FP, TN, FN)

def evaluation(TP, FP, TN, FN):
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    print("Accuracy:", Accuracy)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("F1:", F1)
    return Accuracy, Precision, Recall, F1