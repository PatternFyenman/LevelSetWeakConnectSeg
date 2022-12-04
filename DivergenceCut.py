from scipy.ndimage import gaussian_filter, laplace
import pylab as pl
import cv2
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from SurfaceNormal import SurfaceNormal
import time

start = time.time()


img = cv2.imread('./pictures/example1.png', 0)
h, w = 200, 200
img = cv2.resize(img, (w, h))

c0 = 2
phi = c0 * np.ones((h, w))
phi[4: h-4, 4: w-4] = -c0


# hyperparams
mu = 0.2  # mu * timestep < 1/4
timestep = 0.2/ mu
iter_num = 600
lmda = 1.5
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

norm_img = img_smooth / np.max(img_smooth)
sums = []
oldsum = 0.0
for it in range(iter_num):
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
    edge = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * gValue * (nxx + nyy)
    edge_term = lmda * edge

    area_term = dirac_phi * gValue * alfa0
    
    # k = 2
    # gap_term = k * dps * IValue 
    
    phi += timestep * (area_term + edge_term + dist_reg_term)
    

fig.clf()
contours = measure.find_contours(phi, 0)
ax1 = fig.add_subplot(231)
ax1.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
sum_iter = 0.0
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    sum_iter += np.sum(contour)

ax2 = fig.add_subplot(232, projection='3d')
y, x = phi.shape
x = np.arange(0, x, 1)
y = np.arange(0, y, 1)
X, Y = np.meshgrid(x, y)
ax2.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
ax2.contour(X, Y, phi, 0, colors='g', linewidths=2)

ax3 = fig.add_subplot(233)
ax3.imshow(phi)

ax4 = fig.add_subplot(234)

def getCutPoint(Ux, Vy, move=5):
    sig = 1
    div_field = Ux + Vy
    loch, locw = np.where(div_field == np.max(div_field))
    points = []
    for l in range(len(loch)):
        arrow_h = loch[l]
        arrow_w = locw[l]
        arrow_avg_hs = []
        arrow_avg_ws = []
        for lh in range(-sig, sig+1, 1):
            for lw in range(-sig, sig+1, 1):
                arrow_avg_hs.append(Ux[arrow_h + lh, arrow_w + lw])
                arrow_avg_ws.append(Vy[arrow_h + lh, arrow_w + lw])
        arrow_avg_h = np.mean(arrow_avg_hs)
        arrow_avg_w = np.mean(arrow_avg_ws)
        arrow_norm = np.sqrt(arrow_avg_h ** 2 + arrow_avg_w ** 2)
        point_h = arrow_h + move * arrow_avg_h / (arrow_norm + 1e-8)
        point_w = arrow_w + move * arrow_avg_w / (arrow_norm + 1e-8)
        if 0 < point_h < h and 0 < point_w < w:
            points.append([point_h, point_w])

    return points

X = [xi for xi in range(h)]
Y = [yi for yi in range(w)]
U, V = SurfaceNormal(phi)
div_field = np.zeros_like(U)
[_, Ux] = np.gradient(U)
[Vy, _] = np.gradient(V) 
div_field = Ux + Vy
ax4.imshow(div_field)
points = getCutPoint(Ux, Vy, move=10)
if points:
    print("get cutting points!")
    for p in points:
        ax4.scatter(p[0], p[1], marker='o', c='r')


ax5 = fig.add_subplot(235)
H = [hi for hi in range(h)]
W = [wi for wi in range(w)]
H, W = np.meshgrid(H, W)
ax5.quiver(W, H, Vy[W, H], Ux[W, H], units="xy")

def GaussianField(center_x, center_y, sigma):
    normal = 1 / (2 * 3.1415926 * sigma ** 2)
    hh = np.arange(0, h, 1)
    ww = np.arange(0, w, 1)
    H, W = np.meshgrid(hh, ww)
    dst = np.sqrt((H-center_y)**2 + (W-center_x)**2)
    gfield = normal * np.exp(-1 * dst / (2.0 * sigma ** 2))
    gfield = gfield / np.max(gfield)
    return gfield

if points:
    for p in points:
        GField = GaussianField(p[0], p[1], 2)
        phi += GField * 2 * c0

ax6 = fig.add_subplot(236)
ax6.imshow(phi)

plt.pause(1)
# end = time.time()
# print("time consuming: ", end - time1, " seconds")

for it in range(300):
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
    edge = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * gValue * (nxx + nyy)
    edge_term = lmda * edge

    area_term = dirac_phi * gValue * alfa0
    
    # k = 2
    # gap_term = k * dps * IValue 
    
    phi += timestep * (area_term + edge_term + dist_reg_term)
    if it % 20 == 0:
        fig.clf()
        contours = measure.find_contours(phi, 0)
        ax1 = fig.add_subplot(231)
        ax1.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        sum_iter = 0.0
        for n, contour in enumerate(contours):
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
            sum_iter += np.sum(contour)

        ax2 = fig.add_subplot(232, projection='3d')
        y, x = phi.shape
        x = np.arange(0, x, 1)
        y = np.arange(0, y, 1)
        X, Y = np.meshgrid(x, y)
        ax2.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
        ax2.contour(X, Y, phi, 0, colors='g', linewidths=2)

        plt.pause(0.3)
