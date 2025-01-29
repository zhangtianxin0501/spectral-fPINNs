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
from visualization import plot_u1,plot_u2, plot_u3, plot_x, plot_u4
from spe_t_all import exact, Sequentialmodel, training_data_lhb

x_l = -1
x_r = 1
N_xdata = 100

t_0 = 0
t_T = 0.1
N_tdata = 100

x = np.linspace(x_l, x_r, N_xdata) # x_r = 4
t = np.linspace(t_0, t_T, N_tdata)
X , T = np.meshgrid(x,t)
print("X：",X.shape)
print("T：",T.shape)
U_gt = exact(X,T).T
print("size of (x, t):", U_gt.shape)
# plot_u(U_gt, x, t)

# test dataset
x_t_test = np.vstack((X.flatten(), T.flatten())).T
u_test = U_gt.flatten('F')[:,None]

# training dataset

N_boundary = 5
N_inner = 30

lb = np.array([x_l,t_0])
ub = np.array([x_r,t_T])
x_t_train, x_t_boundary, u_boundary = training_data_lhb(X, T, lb, ub, U_gt, N_boundary, N_inner)

# Device configuration
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

if device == 'mps':
    print(torch.cuda.get_device_name())

layers = np.array([2,20,20,20,20,1]) #4 hidden layers
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

optimizer = torch.optim.Adam(PINN.parameters(), lr = 0.0001)

start = timer()
PINN.train_model_adam(optimizer, x_t_boundary, u_boundary, x_t_train, 20000)
end = timer()
print("consumed time: " + str(end - start) + "s")

error_vec, u_pred = PINN.test(x_t_test, u_test, N_xdata, N_tdata)
print('Test Error: %.5f'  % (error_vec))
plot_u2(u_pred,x,t)
plot_u1(U_gt,x,t)
error = abs(u_pred-U_gt)
plot_u3(error,x,t)
#plot_u_x(u_pred, U_gt, x,t, pos=[0,50,99])
plot_u4(u_pred, U_gt,x)
#plt.savefig("result.jpg")
plt.show()