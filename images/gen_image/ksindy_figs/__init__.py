# %%
from typing import NewType

import kalman
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.linear_model import Lasso

NTimesteps = NewType("NTimesteps", int)
NSeries = NewType("NSeries", int)
Float1D = np.ndarray[NTimesteps, np.dtype[np.floating]]

CMAP = mpl.color_sequences['tab10']
np.random.seed(1)
NOISE_VAR = .2

f_dot = lambda t: .3 * t **2 - 2 * t
f = lambda t: 20 / (3 + 2 * np.e ** (2 * t))

def generate_mock_data(dt: float, noise_var: float) -> tuple[float, Float1D, Float1D, Float1D, Float1D]:
    t = np.arange(0, 6, dt)
    nt = len(t)
    x = np.sin(t)
    x_dot = np.cos(t)
    z = x + np.random.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z
def alt_mock_data(dt: float, noise_var: float) -> tuple[float, Float1D, Float1D, Float1D, Float1D]:
    t = np.arange(0, 6, dt)
    nt = len(t)
    x = f(t)
    x_dot = f_dot(t)
    z = x + np.random.normal(size=x.shape, scale=np.sqrt(noise_var))
    return nt, t, x, x_dot, z

# %%
nt, t, x, x_dot, z = generate_mock_data(.3, NOISE_VAR)
nt, t, x, x_dot, z = alt_mock_data(.3, NOISE_VAR)
# %%
zoom_start = len(t) //3
zoom_inds = slice(zoom_start, zoom_start+10)
plt.plot(t[zoom_inds], z[zoom_inds], ".")

# %%
H = sparse.lil_matrix((nt, 2 * nt))
H[:, 1::2] = sparse.eye(nt)

x_hat, x_dot_hat, G, Qinv, loss = kalman.solve(z, H, t, NOISE_VAR, 1e-1)

# %%

plt.plot(t, x, label="true")
plt.plot(t, z, ".", label="measured")
plt.plot(t, x_hat, label="kalman")
plt.legend()

# %%
def fd_arrows(
    x: Float1D,
    y: Float1D
) -> np.ndarray[tuple[NTimesteps, NSeries], np.dtype[np.floating]]:
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return np.stack((dx, dy), axis=0)

# %%
fd = fd_arrows(t, x)[:, zoom_inds]
zd = fd_arrows(t, z)[:, zoom_inds]


# %%
q_props = {"headwidth": 1.5, "headlength": 2, }
# plt.plot(t[zoom_inds], x[zoom_inds], label="true")
plt.title("Kalman smoothing applied to measurements")
plt.plot(t[zoom_inds], z[zoom_inds], ".", color=CMAP[0], label="measured")
plt.plot(t[zoom_inds], x_hat[zoom_inds], "--", color=CMAP[1], label="kalman")
plt.quiver(t[zoom_inds], z[zoom_inds], zd[0], zd[1], color=CMAP[0], **q_props)
plt.quiver(t[zoom_inds], x_hat[zoom_inds], zd[0], x_dot_hat[zoom_inds]*zd[0], color=CMAP[1], **q_props)
# plt.plot(t[zoom_inds], x_dot_hat[zoom_inds], color=cmap[1])
plt.legend()

# %%
funcs_theta = np.stack((np.ones_like(x_hat), x_hat, x_hat**2), axis=0)
xyu = (t[zoom_inds], x_hat[zoom_inds], zd[0])
plt.plot(t[zoom_inds], x_hat[zoom_inds], ".", color=CMAP[1], label="kalman")
plt.quiver(*xyu, x_dot_hat[zoom_inds]*zd[0], color=CMAP[1])
plt.quiver(*xyu, funcs_theta[0, zoom_inds]*zd[0], **q_props)
plt.quiver(*xyu, funcs_theta[1, zoom_inds]*zd[0], **q_props)
plt.quiver(*xyu, funcs_theta[2, zoom_inds]*zd[0], **q_props)
# %%

model = Lasso(1e-1, fit_intercept=False).fit(funcs_theta.T, x_dot_hat)
model.coef_
