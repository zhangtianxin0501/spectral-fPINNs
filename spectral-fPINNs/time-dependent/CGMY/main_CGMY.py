import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io
from timeit import default_timer as timer

# My Funs
from visualization import plot_u, plot_x, plot_u_x
from CGMY import  Sequentialmodel, training_data_lhb

from scipy.io import loadmat

x_l = -1
x_r = 1
N_xdata = 2**9

t_0 = 0
t_T = 0.1
N_tdata = 20

x = np.linspace(x_l, x_r, N_xdata+1)
#print("x",x)
t = np.linspace(t_0, t_T, N_tdata)
#tao = t_T - t
X , T = np.meshgrid(x,t)
#print(X)
print("X：",X.shape)
print("T：",T.shape)
#U_gt = exact(X,T).T
#print("size of (x, t):", U_gt.shape)
# plot_u(U_gt, x, t)

# test dataset
x_t_test = np.vstack((X.flatten(), T.flatten())).T
#x_t_test = np.vstack((np.exp(X[0].flatten()), np.ones_like(X[0].flatten()))).T
#print("x_t_test",x_t_test)
#u_test = U_gt.flatten('F')[:,None]
#x_t_test2 = np.reshape(x_t_test[:,-1], (N_xdata, N_tdata), order='F')
#print("x_t_test2",x_t_test2)
# training dataset

N_boundary = 90
N_inner = 60



lb = np.array([x_l,t_0])
ub = np.array([x_r,t_T])
x_t_train, x_t_boundary, u_boundary = training_data_lhb(X, T, lb, ub,  N_boundary, N_inner)

# Device configuration
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

if device == 'mps':
    print(torch.cuda.get_device_name())

layers = np.array([2,30,30,30,30,1]) #4 hidden layers
PINN = Sequentialmodel(layers, device)

PINN.to(device)
# print(PINN)


"""
optimizer = torch.optim.LBFGS(PINN.parameters(), lr=0.0001,
                              max_iter = 10000,
                              max_eval = 200,
                              tolerance_grad = 1e-09,
                              tolerance_change = 1e-16,
                              history_size = 100,
                              line_search_fn = 'strong_wolfe')
PINN.train_model(optimizer, x_t_boundary, u_boundary, x_t_train) # BFGS
"""

optimizer = torch.optim.Adam(PINN.parameters(), lr = 1e-4)

start = timer()
PINN.train_model_adam(optimizer, x_t_boundary, u_boundary, x_t_train, 30000)
end = timer()
print("consumed time: " + str(end - start) + "s")

#Ve_dict = loadmat("Ve3.mat")
Ve_dict = loadmat("Ve.mat")
Ve = Ve_dict["Ve"]
#print("Ve",Ve)
plt.plot(x[1:-1],Ve,label='Reference',color='b',linewidth=2.5)

#error_vec, u_pred = PINN.test(x_t_test, u_test, N_xdata, N_tdata=1)
u_pred = PINN.test1(x_t_test, N_xdata+1, N_tdata)
#print('Test Error: %.5f'  % (error_vec))
#plot_u_x(u_pred, U_gt, x,t, pos=[-1])
print("u_pred",u_pred)
#first_elements = [row[0] for row in u_pred][1:-1]
last_elements = [row[-1] for row in u_pred][1:-1]
print("last_elements",last_elements)
plt.plot(x[1:-1],last_elements,label='Prediction',ls="--",color='r',linewidth=2)
#plt.plot(x[1:-1],first_elements,label='Intial')
#plt.plot(x[1:-1],np.maximum(1 - np.exp(x[1:-1]), 0),color='r',linewidth=2 ,label='Intial_real')
#plt.xlabel('x')
#plt.ylabel('V')
plt.legend()
#print("Ve",Ve)
last_elements = np.vstack(last_elements)
#print("last_elements",last_elements)
error_vec = np.linalg.norm((Ve - last_elements), 2) / np.linalg.norm(Ve, 2)
print('Test Error: %.5f'  % (error_vec))

plt.savefig("result_CGMY.jpg")
plt.show()