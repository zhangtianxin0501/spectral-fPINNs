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


def exact(x, t):
    U = np.exp(-x) * np.cos(t)
    return U


def training_data_lhb(X, T, lb, ub, U_gt, N_boundary=20, N_inner=200):
    '''Boundary Conditions'''

    # Initial Condition lb[0] =< x =< ub[0] and t = lb[1]
    x_t_IC = np.vstack((X[0, :], T[0, :])).T
    u_IC = U_gt[:, 0][:, None]

    # Boundary Condition x = x_l and lb[1] =< t =< ub[1]
    x_t_BCl = np.vstack((X[:, 0], T[:, 0])).T
    u_BCl = U_gt[0, :][:, None]

    # Boundary Condition x = x_r and lb[1] =< t =< ub[1]
    x_t_BCr = np.vstack((X[:, -1], T[:, -1])).T
    u_BCr = U_gt[-1, :][:, None]

    # total Boundary
    x_t_boundary = np.vstack([x_t_IC, x_t_BCl, x_t_BCr])
    u_boundary = np.vstack([u_IC, u_BCl, u_BCr])

    # choose random N_u points for training
    idx = np.random.choice(x_t_boundary.shape[0], N_boundary, replace=False)
    x_t_boundary = x_t_boundary[idx, :]  # choose indices from  set 'idx' (x,t)
    u_boundary = u_boundary[idx, :]  # choose corresponding u

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
        self.G = 2
        self.N = 16  # Legendre基底个数
        self.N_G = 16  # Gauss-Legendre求积公式节点数
        self.alpha1 = 1.5
        self.s = 2

        # x_k 和 w_k是Gauss-Legendre求积公式的节点和权重
        self.x_k, self.w_k = scipy.special.roots_legendre(self.N_G + 1)

        self.w_k, self.x_k = torch.from_numpy(self.w_k).float(), torch.reshape(torch.from_numpy(self.x_k).float(),
                                                                               [self.N_G + 1, -1])
        self.L = torch.reshape(self.jacobi_polynomial(0, 0, self.x_k), [-1, self.N_G + 1])
        self.W = torch.diag(self.w_k) * torch.diag(torch.exp( self.G  * self.x_k ).squeeze(1))


        coe1 = []
        for j in range(self.N + 1):
            temp = torch.tensor((2 * j + 1) / 2)
            coe1.append(temp)
        coe1 = torch.vstack(coe1)
        # print("coe1",coe1)
        self.Coeff = self.L @ self.W * coe1
        # print("self.Coeff:", self.Coeff)

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
            # print("z:",z)
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
            # print("jacobi",jacobi)
        jacobi = torch.vstack(jacobi)
        return jacobi


    def fra_term(self, x):

        fra_jacobi = torch.reshape(self.jacobi_polynomial(self.alpha1, 1 - self.alpha1, x),[self.N+1,-1])[:self.N,:]

        Dl_n_result = self.coe2  * fra_jacobi @ torch.diag(torch.pow(1 + x, 1 - self.alpha1).squeeze(1))

        Dl_n_result = torch.flip(Dl_n_result, dims=[0])

        return Dl_n_result

    # 系数a_j
    def a_j(self, t):
        """
        a_j = sum(w_k * L_j(x_k) * e^{G*x_k} * f(x_k))
        """

        x_k_t = torch.cartesian_prod(self.x_k.squeeze(1), t.squeeze(1))


        #a_result  = self.Coeff @ self.forward(self.x_k)
        a_result = self.Coeff @ torch.reshape(self.forward(x_k_t),[self.N_G+1,-1])
        #print("a_result",a_result)

        A = [a_result[self.N]]
        for i in range(self.N):
            temp =a_result[self.N-1-i] - A[i]
            A.append(temp)
        A_result = torch.vstack(A)



        return A_result

    def compute(self, x_t):
        x, t = x_t[:, 0][:, None], x_t[:, 1][:, None]

        A = self.a_j(t)

        B = self.fra_term(x)

        C = A[:-1] *B

        C_sum = torch.sum(C, 0).unsqueeze(1)

        C0 = A[-1].unsqueeze(1) * self.coe3 * pow(1 + x, self.s - self.alpha1 - 2)

        result = torch.exp(-self.G * x)*(C_sum + C0)


        return result

    'non-homogeneous term for Loss'

    def fun_f(self, x_t):

        x, t = x_t[:, 0][:, None], x_t[:, 1][:, None]
        a = self.G - 1
        k = self.alpha1 - self.s + 1

        m = 1 - k
        n = torch.mul(a, 1 + x)

        max_iter = 10  # 最大迭代次数
        result = 0.0  # 结果初始化
        mul = 1
        for _ in range(max_iter):
            mul *= m + _
            # print("mul:",mul)
            term = torch.pow(n, _) / mul
            result += term
        result1 = math.gamma(m) - torch.pow(n, m) * torch.exp(-n) * result

        temp = (self.G - 1) ** self.alpha1 * torch.exp(-x) * result1 - \
               (self.G - 1) * torch.exp(-self.G * x - self.G + 1) * torch.pow(1 + x, (-self.alpha1 + 1)) + \
               (self.alpha1 - 1) * torch.exp(-self.G * x - self.G + 1) * torch.pow(1 + x, -self.alpha1)
        left = (self.G - 1) ** self.alpha1 * torch.exp(-x) - 1 / math.gamma(self.s - self.alpha1) * temp

        f = -torch.exp(-x)*torch.sin(t) -(left - pow(self.G, self.alpha1) * torch.exp(-x))*torch.cos(t)

        return f

    'Eq term for Loss'

    def PDE(self, x_t):
        g = x_t
        g.requires_grad = True
        V = self.forward(g)

        V_x_t = autograd.grad(V, g, torch.ones([x_t.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        V_t = V_x_t[:, [1]]

        f = self.fun_f(x_t)

        D_l = self.compute(x_t)
        L_v =  V_t - D_l + pow(self.G, self.alpha1) * V - f

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

        return  loss_u + loss_L_v

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



        print("error_vec:", error_vec)

        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred, (N_xdata, N_tdata), order='F')

        return error_vec, u_pred