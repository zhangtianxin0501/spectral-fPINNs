import torch
import torch.autograd as autograd  # computation graph
from torch import Tensor  # tensor node in the computation graph
import torch.nn as nn  # neural networks
import torch.optim as optim  # optimizers e.g. gradient descent, ADAM, etc.

import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling
import scipy.io
import matplotlib.pyplot as plt
import math
import scipy.special
import itertools


def training_data_lhb(X, T, lb, ub, N_boundary, N_inner):
    '''Boundary Conditions'''
    K = 1.0

    # Initial Condition lb[0] =< x =< ub[0] and t = lb[1]
    x_t_IC = np.vstack((X[0, :], T[0, :])).T
    #print("x_t_IC:",x_t_IC)
    u_IC = K - np.exp(X[0, :])[:, None]

    u_IC = np.maximum(u_IC, 0)


    # Boundary Condition x = x_l and lb[1] =< t =< ub[1]
    x_t_BCl = np.vstack((X[:, 0], T[:, 0])).T

    u_BCl =  np.zeros(len(x_t_BCl))[:, None]


    # Boundary Condition x = x_r and lb[1] =< t =< ub[1]
    x_t_BCr = np.vstack((X[:, -1], T[:, -1])).T
    u_BCr  = np.zeros(len(x_t_BCr))[:,None]


    # choose random N_u points for training


    idx_IC=np.random.choice(x_t_IC.shape[0],  N_boundary - 16, replace=False)
    x_t_boundary_IC = x_t_IC[idx_IC, :]  # choose indices from  set 'idx' (x,t)
    u_boundary_IC = u_IC[idx_IC, :]  # choose corresponding u

    idx_BCl = np.random.choice(x_t_BCl.shape[0], 8, replace=False)
    x_t_boundary_BCl = x_t_BCl[idx_BCl, :]  # choose indices from  set 'idx' (x,t)
    u_boundary_BCl = u_BCl[idx_BCl, :]  # choose corresponding u

    idx_BCr = np.random.choice(x_t_BCr.shape[0], 8, replace=False)
    x_t_boundary_BCr = x_t_BCr[idx_BCr, :]  # choose indices from  set 'idx' (x,t)
    u_boundary_BCr = u_BCr[idx_BCr, :]  # choose corresponding u

    x_t_boundary = np.vstack([x_t_boundary_IC, x_t_boundary_BCl, x_t_boundary_BCr])
    u_boundary = np.vstack([u_boundary_IC, u_boundary_BCl, u_boundary_BCr])

    num=5
    x_t_Tboundary = np.repeat(np.vstack([[-0.99609375,0.1],[0.99609375,0.1]]),num,axis=0)
    ##Ve3 2^9
    #u_Tboundary = np.repeat(np.vstack([[0.0449],[0.0191]]),num,axis=0)
    #u_Tboundary = np.repeat(np.vstack([[0.160115793021220], [0.0301419828946861]]), num, axis=0)
    #x_t_Tboundary = np.repeat(np.vstack([[-0.9990234375, 0.1], [0.9990234375, 0.1]]), num, axis=0)
    #u_Tboundary = np.repeat(np.vstack([[0.003302213826302], [4.687056583357100e-06]]), num, axis=0)
    ##Ve1 2^9
    u_Tboundary = np.repeat(np.vstack([[0.00936958119705996],[1.31855130368514e-05]]),num,axis=0)
    #print("x_t_Tboundary",x_t_Tboundary)
    x_t_boundary = np.vstack([x_t_boundary,x_t_Tboundary])
    u_boundary = np.vstack([u_boundary,u_Tboundary])




    '''Collocation Points'''

    # latin_hypercube
    x_t_train = lb + (ub - lb) * lhs(2, int(N_inner))

    return x_t_train, x_t_boundary, u_boundary

class Sequentialmodel(nn.Module):

    def __init__(self, layers, device):
        super().__init__()  # call __init__ from parent class
        self.layers = layers
        self.device = device
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.iter = 0
        self.C = 0.2
        self.G = 2
        self.M = 2
        self.r = 0.1
        self.N = 20  # Legendre基底个数
        self.N_G = 20 # Gauss-Legendre求积公式节点数
        self.alpha1 = 1.5
        self.s = 2

        self.d = self.C * math.gamma(-self.alpha1)
        self.b = self.r - self.d * (pow(self.G + 1, self.alpha1) - pow(self.G, self.alpha1) + \
                          pow(self.M - 1, self.alpha1) - pow(self.M, self.alpha1))

        # x_k 和 w_k是Gauss-Legendre求积公式的节点和权重
        self.x_k, self.w_k = scipy.special.roots_legendre(self.N_G + 1)
        self.w_k, self.x_k = torch.from_numpy(self.w_k).float(), torch.reshape(torch.from_numpy(self.x_k).float(),
                                                                               [self.N_G + 1, -1])

        self.L = torch.reshape(self.jacobi_polynomial(0, 0, self.x_k), [-1, self.N_G + 1])

        self.W_l = torch.diag(self.w_k) * torch.diag(torch.exp(self.G * self.x_k).squeeze(1))
        self.W_r = torch.diag(self.w_k) * torch.diag(torch.exp(- self.M * self.x_k).squeeze(1))

        coe1 = []
        for j in range(self.N + 1):
            temp = torch.tensor((2 * j + 1) / 2)
            coe1.append(temp)
        coe1 = torch.vstack(coe1)
        # print("coe1",coe1)
        self.Coeff_l = self.L @ self.W_l * coe1
        self.Coeff_r = self.L @ self.W_r * coe1

        coe2 = []
        for j in range(self.N ):
            temp = torch.tensor(math.gamma(j + 2) / math.gamma(j - self.alpha1 + 2))
            coe2.append(temp)
        self.coe2 = torch.vstack(coe2)
        #print("coe2:",self.coe2)

        self.coe3 = (1 / math.gamma(self.s - self.alpha1) * (self.s - self.alpha1 - 1))

        'Xavier Normal Initialization'
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    'foward pass'

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)

        for i in range(len(self.layers) - 2):
            z = self.linears[i](x)

            x = self.activation(z)

        output = self.linears[-1](x)
        # print("output:",output)
        return output


    def jacobi_polynomial(self, a, b, x):
        jacobi = [torch.ones(x.shape)]
        p0 = 1
        p1 = 1 / 2 * (a + b + 2) * x + 1 / 2 * (a - b)
        jacobi.append(p1)
        for i in range(1, self.N):
            A = (2 * i + a + b + 1) * (2 * i + a + b + 2) / (2 * (i + 1) * (i + a + b + 1))
            B = (b ** 2 - a ** 2) * (2 * i + a + b + 1) / (2 * (i + 1) * (i + a + b + 1) * (2 * i + a + b))
            C = (i + a) * (i + b) * (2 * i + a + b + 2) / ((i + 1) * (i + a + b + 1) * (2 * i + a + b))
            p = (A * x - B) * p1 - C * p0
            p0 = p1
            p1 = p
            jacobi.append(p1)

        jacobi = torch.vstack(jacobi)
        return jacobi

    def fra_term(self, x):

        fra_jacobi_l = torch.reshape(self.jacobi_polynomial(self.alpha1, 1 - self.alpha1, x), [self.N + 1, -1])[:self.N,:]
        Dl_n_result = self.coe2 * fra_jacobi_l @ torch.diag(torch.pow(1 + x, 1 - self.alpha1).squeeze(1))
        Dl_n_result = torch.flip(Dl_n_result, dims=[0])

        fra_jacobi_r = torch.reshape(self.jacobi_polynomial(1 - self.alpha1, self.alpha1, x), [self.N + 1, -1])[:self.N,:]
        Dr_n_result = self.coe2 * fra_jacobi_r @ torch.diag(torch.pow(1 - x, 1 - self.alpha1).squeeze(1))
        Dr_n_result = torch.flip(Dr_n_result, dims=[0])

        return Dl_n_result, Dr_n_result

    # 系数a_j
    def a_j(self, t):
        """
        a_j = sum(w_k * L_j(x_k) * e^{G*x_k} * f(x_k))
        """

        x_k_t = torch.cartesian_prod(self.x_k.squeeze(1), t.squeeze(1))

        a_l = self.Coeff_l @ torch.reshape(self.forward(x_k_t),[self.N_G + 1,-1])
        a_r = self.Coeff_r @ torch.reshape(self.forward(x_k_t), [self.N_G + 1, -1])


        A_l = [a_l[self.N]]
        A_r = [-a_r[self.N]]
        for i in range(self.N):
            temp_l = a_l[self.N - 1 - i] - A_l[i]
            A_l.append(temp_l)
            temp_r = - a_r[self.N - 1 - i] + A_r[i]
            A_r.append(temp_r)
        Al_result = torch.vstack(A_l)
        Ar_result = torch.vstack(A_r)


        return Al_result, Ar_result

    def compute(self, x_t):
        x, t = x_t[:, 0][:, None], x_t[:, 1][:, None]

        Al , Ar = self.a_j(t)

        Bl , Br = self.fra_term(x)

        Cl = Al[:-1] * Bl
        Cl_sum = torch.sum(Cl, 0).unsqueeze(1)
        Cl0 = Al[-1].unsqueeze(1) * self.coe3 * pow(1 + x, self.s - self.alpha1 - 2)
        result_l = torch.exp(-self.G * x)*(Cl_sum + Cl0)

        Cr = Ar[:-1] * Br
        Cr_sum = torch.sum(Cr, 0).unsqueeze(1)
        Cr0 = -Ar[-1].unsqueeze(1) * self.coe3 * pow(1 - x, self.s - self.alpha1 - 2)
        result_r = torch.exp(self.M * x) * (Cr_sum + Cr0)
        return result_l,result_r



    'Eq term for Loss'

    def PDE(self, x_t):
        g = x_t
        g.requires_grad = True
        V = self.forward(g)

        V_x_t = autograd.grad(V, g, torch.ones([x_t.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        V_x = V_x_t[:, [0]]
        V_t = V_x_t[:, [1]]



        D_l , D_r = self.compute(x_t)

        L_v = V_t - self.d * (D_r + D_l - pow(self.M, self.alpha1) * V - pow(self.G, self.alpha1) * V) + \
              self.r * V - self.b * V_x

        return L_v

    def loss(self, x_t_boundary, u_boundary, x_t):

        if torch.is_tensor(x_t) != True:
            x_t = torch.from_numpy(x_t).float().to(self.device)  # 把数组转换成张量，且二者共享内存

        if torch.is_tensor(x_t_boundary) != True:
            x_t_boundary = torch.from_numpy(x_t_boundary).float().to(self.device)

        if torch.is_tensor(u_boundary) != True:
            u_boundary = torch.from_numpy(u_boundary).float().to(self.device)

        loss_u = self.loss_function(self.forward(x_t_boundary), u_boundary)
        L_v = self.PDE(x_t)
        loss_L_v = self.loss_function(L_v, torch.zeros(L_v.shape).to(self.device))

        return  10 * loss_u + loss_L_v

    def train_model_adam(self, optimizer, x_t_boundary, u_boundary, x_t_train, n_epoch):

        while self.iter < n_epoch:
            optimizer.zero_grad()
            loss = self.loss(x_t_boundary, u_boundary, x_t_train)
            loss.backward()

            if self.iter % 1000 == 0:
                print(self.iter, loss)

            self.iter += 1

            optimizer.step()

    'test neural network'

    def test(self, x_t, u, N_xdata, N_tdata):

        if torch.is_tensor(u) != True:
            u = torch.from_numpy(u).float().to(self.device)

        u_pred = self.forward(x_t)
        error_vec = torch.linalg.norm((u - u_pred), 2) / torch.linalg.norm(u, 2)

        # a = u-u_pred

        print("error_vec:", error_vec)

        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred, (N_xdata, N_tdata), order='F')

        return error_vec, u_pred

    def test1(self, x_t,  N_xdata, N_tdata):


        u_pred = self.forward(x_t)


        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred, (N_xdata, N_tdata), order='F')



        return  u_pred