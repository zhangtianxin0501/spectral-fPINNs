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


def exact(x):
    U = np.exp(-x)
    return U


def training_data_lhb(x, lb, ub, U_gt, N_inner=200):
    '''Boundary Conditions'''

    '''Collocation Points'''

    # latin_hypercube
    x_train = lb + (ub - lb) * lhs(1, int(N_inner))

    x_boundary = np.vstack((x[0], x[-1]))
    u_boundary = np.vstack((U_gt[0], U_gt[-1]))

    return x_train, x_boundary, u_boundary


class Sequentialmodel(nn.Module):

    def __init__(self, layers, device):
        super().__init__()  # call __init__ from parent class
        self.layers = layers
        self.device = device
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.iter = 0
        self.M = 2
        self.N = 16  # Legendre基底个数
        self.N_G = 16  # Gauss-Legendre求积公式节点数
        self.alpha1 = 1.5
        self.s = 2

        # x_k 和 w_k是Gauss-Legendre求积公式的节点和权重
        x_k, w_k = scipy.special.roots_legendre(self.N_G + 1)

        self.w_k, self.x_k = torch.from_numpy(w_k).float(), torch.reshape(torch.from_numpy(x_k).float(),
                                                                               [self.N_G + 1, -1])
        self.L = torch.reshape(self.jacobi_polynomial(0, 0, self.x_k), [-1, self.N_G + 1])
        self.W = torch.diag(self.w_k) * torch.diag(torch.exp( - self.M * self.x_k).squeeze(1))


        coe1 = []
        for j in range(self.N + 1):
            temp = torch.tensor((2 * j + 1) / 2)
            coe1.append(temp)
        coe1 = torch.vstack(coe1)

        self.Coeff = self.L @ self.W * coe1


        coe2 = []
        for j in range(self.N ):
            temp = torch.tensor(math.gamma(j + 2) / math.gamma(j - self.alpha1 + 2))
            coe2.append(temp)
        self.coe2 = torch.vstack(coe2)


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



    def fra_term2(self, x):

        fra_jacobi2 = torch.reshape(self.jacobi_polynomial( 1 - self.alpha1,self.alpha1, x),[self.N+1,-1])[:self.N,:]

        Dr_n_result = self.coe2  * fra_jacobi2 @ torch.diag(torch.pow(1 - x, 1 - self.alpha1).squeeze(1))

        Dr_n_result = torch.flip(Dr_n_result, dims=[0])

        return Dr_n_result

    # 系数a_j
    def a_j(self):
        """
        a_j = sum(w_k * L_j(x_k) * e^{-M*x_k} * f(x_k))
        """

        a_result  = self.Coeff @ self.forward(self.x_k)
        #print("a_result",a_result)
        A = [-a_result[self.N]]
        for i in range(self.N):
            temp =- a_result[self.N-1-i] + A[i]
            A.append(temp)

        A_result = torch.vstack(A)


        return A_result

    def compute(self, x):

        A = self.a_j()

        B = self.fra_term2(x)

        C = A[:-1] *B

        C_sum = torch.sum(C, 0).unsqueeze(1)

        C0 = -A[-1] * self.coe3 * pow(1 - x, self.s - self.alpha1 - 2)

        result = torch.exp(self.M * x) * (C_sum + C0)


        return result

    'non-homogeneous term for Loss'

    def fun_f(self, x):
        # print("x:",x)

        a = self.M + 1
        k = self.alpha1 - self.s + 1

        m = 1 - k
        n = torch.mul(a, 1 - x)

        max_iter = 20  # 最大迭代次数
        result = 0.0  # 结果初始化
        mul = 1
        for _ in range(max_iter):
            mul *= m + _

            term = torch.pow(n, _) / mul
            result += term
        result1 = math.gamma(m) - torch.pow(n, m) * torch.exp(-n) * result

        temp = a ** self.alpha1 * torch.exp(-x) * result1 - \
               a * torch.exp(self.M * x - self.M - 1) * torch.pow(1 - x, (-self.alpha1 + 1)) + \
               (self.alpha1 - 1) * torch.exp(self.M * x - self.M - 1) * torch.pow(1 - x, -self.alpha1)
        right = a ** self.alpha1 * torch.exp(-x) - 1 / math.gamma(self.s - self.alpha1) * temp

        f =  right - pow(self.M, self.alpha1) * torch.exp(-x)

        return f

    'Eq term for Loss'

    def PDE(self, x):
        g = x
        g.requires_grad = True
        V = self.forward(g)

        D_r = self.compute(x)

        f = self.fun_f(x)

        L_v = D_r - pow(self.M, self.alpha1) * V - f


        return L_v

    def loss(self, x_boundary, u_boundary, x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x).float().to(self.device)  # 把数组转换成张量，且二者共享内存

        if torch.is_tensor(x_boundary) != True:
            x_boundary = torch.from_numpy(x_boundary).float().to(self.device)

        if torch.is_tensor(u_boundary) != True:
            u_boundary = torch.from_numpy(u_boundary).float().to(self.device)

        loss_u = self.loss_function(self.forward(x_boundary), u_boundary)
        L_v = self.PDE(x)
        loss_L_v = self.loss_function(L_v, torch.zeros(L_v.shape).to(self.device))

        return  loss_u + loss_L_v

    def train_model_adam(self, optimizer, x_boundary, u_boundary, x_train, n_epoch):

        while self.iter < n_epoch:
            optimizer.zero_grad()
            loss = self.loss(x_boundary, u_boundary, x_train)
            loss.backward()

            if self.iter % 1000 == 0:
                print(self.iter, loss)

            self.iter += 1

            optimizer.step()

    'test neural network'

    def test(self, x, u, N_xdata):
        # print("x:",x)
        x ,u= np.reshape(x, [-1, 1]),np.reshape(u,[-1,1])
        if torch.is_tensor(u) != True:
            u = torch.from_numpy(u).float().to(self.device)

        u_pred = self.forward(x)

        error_vec = torch.linalg.norm((u - u_pred), 2) / torch.linalg.norm(u, 2)

        # a = u-u_pred

        print("error_vec:", error_vec)

        u_pred = u_pred.cpu().detach().numpy()
        # u_pred = np.reshape(u_pred,(N_xdata, N_tdata), order='F')

        return error_vec, u_pred