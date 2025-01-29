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
from spectral_all import exact, Sequentialmodel, training_data_lhb

x_l = -1
x_r = 1
N_xdata = 1000



x = np.linspace(x_l, x_r, N_xdata)



#X , T = np.meshgrid(x,t)
U_gt = exact(x)



# training dataset


N_inner = 20

lb = np.array([x_l])
ub = np.array([x_r])
x_train, x_boundary, u_boundary = training_data_lhb(x, lb, ub, U_gt,  N_inner)



# Device configuration
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

if device == 'mps':
    print(torch.cuda.get_device_name())

layers = np.array([1,20,20,20,20,1]) #4 hidden layers
PINN = Sequentialmodel(layers, device)

PINN.to(device)



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
PINN.train_model_adam(optimizer, x_boundary, u_boundary, x_train, 20000)
end = timer()
print("consumed time: " + str(end - start) + "s")

error_vec, u_pred = PINN.test(x, U_gt, N_xdata)
print('Test Error: %.5f'  % (error_vec))

plt.plot(x,U_gt,color='b',label='Exact',linewidth=2.5)
plt.plot(x,u_pred,'--',color='r',label='Prediction',linewidth=2.5)
plt.legend()
plt.xlabel('x')
plt.ylabel('u')

plt.savefig("result.jpg")
plt.show()
