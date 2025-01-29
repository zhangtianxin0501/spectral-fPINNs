import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
import matplotlib.colors as cls
import numpy as np

def plot_u(u, x, t, log=False):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    if not log:
        h = ax.imshow(u, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto')
    else:
        h = ax.imshow(u, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto', norm=cls.LogNorm(vmin=1e-5,vmax=1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Real u(x,t)', fontsize = 10)
    return ax

def plot_x(u, U_gt, x, pos):
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,U_gt.T[pos[0],:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u.T[pos[0],:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.03s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-2.1,2.1])
    #ax.set_ylim([-0.1,4.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,U_gt.T[pos[1],:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u.T[pos[1],:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-2.1, 2.1])
    #ax.set_xlim([-1.1,1.1])
    #ax.set_ylim([-0.1,4.1])
    ax.set_title('$t = 0.07s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,U_gt.T[pos[2],:], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(x,u.T[pos[2],:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-2.1, 2.1])
    #ax.set_xlim([-1.1,1.1])
    #ax.set_ylim([-0.1,4.1])
    ax.set_title('$t = 0.1s$', fontsize = 10)


def plot_u_x(u, U_gt, x, t, pos=[3,6,9]):
    ax = plot_u(u, x,  t)
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[pos[0]]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[pos[1]]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[pos[2]]*np.ones((2,1)), line, 'w-', linewidth = 1)
    plot_x(u, U_gt, x, pos)

def plot_u1(u, x, t, log=False):
    fig, ax = plt.subplots()
    # ax.axis('off')
    if not log:
        h = ax.imshow(u, interpolation='nearest', cmap='jet',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
    else:
        h = ax.imshow(u, interpolation='nearest', cmap='YlGnBu',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto', norm=cls.LogNorm(vmin=1e-5, vmax=1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('Real u(x,t)', fontsize=10)
    return ax

def plot_u2(u, x, t, log=False):
    fig, ax = plt.subplots()
    # ax.axis('off')
    if not log:
        h = ax.imshow(u, interpolation='nearest', cmap='jet',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
    else:
        h = ax.imshow(u, interpolation='nearest', cmap='YlGnBu',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto', norm=cls.LogNorm(vmin=1e-5, vmax=1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('Predicted u(x,t)', fontsize=10)
    return ax


def plot_u3(u, x, t, log=False):
    fig, ax = plt.subplots()
    #ax.axis('off')
    if not log:
        h = ax.imshow(u, interpolation='nearest', cmap='jet',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
    else:
        h = ax.imshow(u, interpolation='nearest', cmap='YlGnBu',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto', norm=cls.LogNorm(vmin=1e-5, vmax=1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('Absolute error', fontsize=10)
    return ax

def plot_u4(u_pred, U_gt,x):
    fig, ax = plt.subplots()
    ax.plot(x, U_gt.T[99,:], color='b', label='Exact', linewidth=2.5)
    ax.plot(x, u_pred.T[99,:], '--', color='r', label='Prediction', linewidth=2.5)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('t=1', fontsize=10)
