'''
date: 2022-12-03
description: manually select a point with the mouse within a fuzzy gap
'''

from tqdm import tqdm
from scipy.ndimage import gaussian_filter, laplace
import pylab as pl
import cv2
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import time
from EulerEquationSolver import diffusion
start = time.time()


img = cv2.imread('./pictures/example1.png', 0)
h, w = 256, 256
img = cv2.resize(img, (w, h))

c0 = 2
phi = c0 * np.ones((h, w))
phi[4: h-4, 4: w-4] = -c0


# hyperparams
mu = 0.2  # mu * timestep < 1/4
timestep = 0.2 / mu
iter_num = 500
lmda = 0.2
alfa0 = 1.5
epsilon = 1.5
sigma = 1.5

img = np.array(img, dtype='float32')
img_smooth = gaussian_filter(img, sigma)
[Iy, Ix] = np.gradient(img_smooth)
f = np.square(Ix) + np.square(Iy)
gValue = 1 / (1 + f)
IValue = 1 / (1 + (img_smooth) ** 2)

plt.ion()
fig = plt.figure()

time1 = time.time()

u, v, n = diffusion(img, weight=False, init='grad')

norm_img = img_smooth / np.max(img_smooth)
sums = []
oldsum = 0.0
print("Shrinking zero level set by DRLSE algorithm")
for it in tqdm(range(iter_num)):
    # start level set evolution
    time_a = time.time()

    phi[np.ix_([0, -1], [0, -1])] = phi[np.ix_([2, -3], [2, -3])]
    phi[np.ix_([0, -1]), 1:-1] = phi[np.ix_([2, -3]), 1:-1]
    phi[1:-1, np.ix_([0, -1])] = phi[1:-1, np.ix_([2, -3])]
    time_b = time.time()

    dfunc = (1 / 2 / epsilon) * (1 + np.cos(np.pi * phi / epsilon))
    conds = (phi <= epsilon) & (phi >= -epsilon)
    dirac_phi = dfunc * conds


    [phi_y, phi_x] = np.gradient(phi)
    s =np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s-1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))

    k_x = dps * phi_x - phi_x
    k_y = dps * phi_y - phi_y
    [_, kxx] = np.gradient(k_x)
    [kyy, _] = np.gradient(k_y)
    dist_reg = kxx + kyy + laplace(phi, mode = 'nearest')
    dist_reg_term = mu * dist_reg


    [vy, vx] = np.gradient(gValue)
    n_x = phi_x / (s + 1e-10)
    n_y = phi_y / (s + 1e-10)
    [_, nxx] = np.gradient(n_x)
    [nyy, _] = np.gradient(n_y)
    Dfield = n_x * u + n_y * v
    [Dy, Dx] = np.gradient(Dfield)
    # edge = dirac_phi * (Dfield * (vx + vy) + gValue * (Dx + Dy))
    edge = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * gValue * (nxx + nyy)
    edge_term = lmda * edge

    area_term = dirac_phi * gValue * alfa0
    
    phi += timestep * (area_term + edge_term + dist_reg_term)
    
time2 = time.time()
print("DRLSE time = ", time2 - time1, "s")# 22.367s
def GaussianField(center_x, center_y, sigma):
    normal = 1 / (2 * 3.1415926 * sigma ** 2)
    hh = np.arange(0, h, 1)
    ww = np.arange(0, w, 1)
    H, W = np.meshgrid(hh, ww)
    dst = np.sqrt((H-center_y)**2 + (W-center_x)**2)
    gfield = normal * np.exp(-1 * dst / (2.0 * sigma ** 2))
    gfield = gfield / np.max(gfield)
    return gfield
point_in_200 = 110
ratio = point_in_200 / 200
Gfield = GaussianField(int(h * ratio), int(w * ratio), 2)
img_min = np.min(img)
img_max = np.max(img)
subinten = int((img_max - img_min) / 4)
mask = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        val =img[i][j]
        if val < img_min + subinten:
            mask[i][j] = 0
        elif val >= img_min + subinten and val < img_min + 2 * subinten:
            mask[i][j] = 1
        elif val >= img_min + 2 * subinten and val < img_min + 3 * subinten:
            mask[i][j] = 2
        else:
            mask[i][j] = 3

prevent = np.exp(-mask/0.5)

phi_G = phi + Gfield * 2 * c0

mu = 0.2  # mu * timestep < 1/4
timestep = 0.2 / mu
lmda = 0.1
iter_num = 600
alfa0 = 1.0
phi = phi_G.copy()
print("Shrinking zero level set after cutting the gap")
for it in tqdm(range(iter_num)):
    # start level set evolution
    time_a = time.time()

    phi[np.ix_([0, -1], [0, -1])] = phi[np.ix_([2, -3], [2, -3])]
    phi[np.ix_([0, -1]), 1:-1] = phi[np.ix_([2, -3]), 1:-1]
    phi[1:-1, np.ix_([0, -1])] = phi[1:-1, np.ix_([2, -3])]
    time_b = time.time()

    dfunc = (1 / 2 / epsilon) * (1 + np.cos(np.pi * phi / epsilon))
    conds = (phi <= epsilon) & (phi >= -epsilon)
    dirac_phi = dfunc * conds


    [phi_y, phi_x] = np.gradient(phi)
    s =np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s-1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))

    k_x = dps * phi_x - phi_x
    k_y = dps * phi_y - phi_y
    [_, kxx] = np.gradient(k_x)
    [kyy, _] = np.gradient(k_y)
    dist_reg = kxx + kyy + laplace(phi, mode = 'nearest')
    dist_reg_term = mu * dist_reg


    [vy, vx] = np.gradient(gValue)
    n_x = phi_x / (s + 1e-10)
    n_y = phi_y / (s + 1e-10)
    [_, nxx] = np.gradient(n_x)
    [nyy, _] = np.gradient(n_y)
    Dfield = n_x * u + n_y * v
    [Dy, Dx] = np.gradient(Dfield)
    #edge = dirac_phi * (Dfield * (vx + vy) + gValue * (Dx + Dy))
    edge = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * gValue * (nxx + nyy)
    edge_term = lmda * edge

    area_term = dirac_phi * gValue * alfa0
    
    phi += prevent * timestep * (area_term + edge_term + dist_reg_term)

    if it % 100 == 0:
        print(it)
        fig.clf()
        contours = measure.find_contours(phi, 0)
        ax1 = fig.add_subplot(121)
        ax1.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        sum_iter = 0.0
        for n, contour in enumerate(contours):
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
            sum_iter += np.sum(contour)
        plt.title("zero level set(2D)")

        sums.append(sum_iter - oldsum)
        #print("iter = ", it, ", diff measure = ", sum_iter - oldsum)
        oldsum = sum_iter

        ax2 = fig.add_subplot(122, projection='3d')
        y, x = phi.shape
        x = np.arange(0, x, 1)
        y = np.arange(0, y, 1)
        X, Y = np.meshgrid(x, y)
        ax2.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
        ax2.contour(X, Y, phi, 0, colors='g', linewidths=2)
        plt.title("level set function(3D)")
        plt.pause(0.01)

end = time.time()
print("time consuming: ", end - start, " seconds")
