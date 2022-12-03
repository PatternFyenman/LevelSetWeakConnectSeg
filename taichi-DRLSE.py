import cv2
from tqdm import tqdm
import time
import numpy as np
import taichi as ti
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

ti.init(arch=ti.cpu, debug=True)

resize_shape = 1024
img2d = ti.types.ndarray()
mu = 0.2
timestep = 0.2 / mu
iter_num = 400
lmda = 1.5
alfa0 = 1.5
epsilon = 1.5
sigma = 1.5


# Update activation field x, according to the difference of the phi
@ti.kernel
def activate(phi_x:img2d, phi_y:img2d):
    shape = phi_x.shape
    for h, w in ti.ndrange(shape[0], shape[1]):
        if abs(phi_y[h, w]) >= 0.5 or abs(phi_x[h, w]) >= 0.5:
            x[h, w] = 1


# Update level set func phi, according to activation field x
@ti.kernel
def update_phi(phi:img2d, dphi:img2d): 
    for h, w in x:
        phi[h, w] += dphi[h, w]


def init_phi(phi):
    phii = phi.copy()
    # Neumann condition
    phii[np.ix_([0, -1], [0, -1])] = phii[np.ix_([2, -3], [2, -3])]
    phii[np.ix_([0, -1]), 1:-1] = phii[np.ix_([2, -3]), 1:-1]
    phii[1:-1, np.ix_([0, -1])] = phii[1:-1, np.ix_([2, -3])]
    return phii

@ti.kernel
def dirac_phi_taichi(phi:img2d, dirac_phi:img2d):
    for h, w in x:
        dfunc = (1 / 2 / epsilon) * (1 + ti.cos(3.14159 * phi[h, w] / epsilon))
        conds = (phi[h, w] <= epsilon) & (phi[h, w] >= -epsilon)
        dirac_phi[h, w] = dfunc * conds


@ti.kernel
def dps_taichi(s:img2d, phi_x:img2d, phi_y:img2d, dps:img2d):
    for h, w in x:
        s[h, w] =ti.sqrt(phi_x[h, w]**2 + phi_y[h, w]**2)
        a = (s[h, w] >= 0) & (s[h, w] <= 1)
        b = (s[h, w] > 1)
        ps = a * ti.sin(2 * 3.14159 * s[h, w]) / (2 * 3.14159) + b * (s[h, w]-1)
        dps[h, w] = ((ps != 0) * ps + (ps == 0)) / ((s[h, w] != 0) * s[h, w] + (s[h, w] == 0))


@ti.kernel
def kx_ky_taichi(phi_x:img2d, phi_y:img2d, dps:img2d, k_x:img2d, k_y:img2d):
    for h, w in x:
        k_x[h, w] = dps[h, w] * phi_x[h, w] - phi_x[h, w]
        k_y[h, w] = dps[h, w] * phi_y[h, w] - phi_y[h, w]

@ti.kernel
def kxx_kyy_taichi(k_x:img2d, k_y:img2d, kxx:img2d, kyy:img2d):
    for h, w in x:
        if 0 < h < height and 0 < w < weight:
            kxx[h, w] = (k_x[h, w + 1] - k_x[h, w - 1]) / 2
            kyy[h, w] = (k_y[h + 1, w] - k_y[h - 1, w]) / 2

@ti.kernel
def laplace(phi:img2d, phi_laplace:img2d):
    for h, w in x:
        if 0 < h < height and 0 < w < weight:
            phi_laplace[h, w] = 4 * phi[h, w] - phi[h-1, w] - phi[h, w-1] - phi[h+1, w] - phi[h, w+1]
        
    
@ti.kernel
def dist_taichi(mu:ti.f32, kxx:img2d, kyy:img2d, laplace:img2d, dist:img2d):
    for h, w in x:
        dist[h, w] = mu * ( kxx[h, w] + kyy[h, w] + laplace[h, w] )

@ti.kernel
def vx_vy_taichi(gValue:img2d, vx:img2d, vy:img2d):
    for h, w in x:
        vx[h, w] = (gValue[h, w+1] - gValue[h, w-1]) / 2
        vy[h, w] = (gValue[h+1, w] - gValue[h-1, w]) / 2

@ti.kernel
def normalize_phi_taichi(phi_d:img2d, s:img2d, n:img2d):
    for h, w in x:
        n[h, w] = phi_d[h, w] / (s[h, w] + 1e-10)
    
@ti.kernel
def nxx_nyy_taichi(n_x:img2d, n_y:img2d, nxx:img2d, nyy:img2d):
    for h, w in x:
        nxx[h, w] = (n_x[h, w+1] - n_x[h, w-1]) / 2
        nyy[h, w] = (n_y[h+1, w] - n_y[h-1, w]) / 2

@ti.kernel
def edge_taichi(lmda:ti.f32, dirac_phi:img2d, gValue:img2d, vx:img2d, vy:img2d, n_x:img2d, n_y:img2d, nxx:img2d, nyy:img2d, edge:img2d):
    for h, w in x:
        edge[h, w] = lmda * dirac_phi[h, w] * (vx[h, w] * n_x[h, w] + vy[h, w] * n_y[h, w] + gValue[h, w] * (nxx[h, w] + nyy[h, w]))


@ti.kernel
def area_taichi(alfa:ti.f32, dirac_phi:img2d, gValue:img2d, area:img2d):
    for h, w in x:
        area[h, w] = alfa * dirac_phi[h, w] * gValue[h, w]


@ti.kernel
def dphi_taichi(area:img2d, dist:img2d, edge:img2d, dphi:img2d):
    for h, w in x:
        dphi[h, w] =  area[h, w] + dist[h, w] + edge[h, w]

@ti.kernel
def update_phi(phi:img2d, dphi:img2d):
    for h, w in x:
        phi[h, w] += timestep * dphi[h, w]


if __name__ == "__main__":

    print("Image Size = ", resize_shape, " x ", resize_shape)
    start = time.time()

    # Initialize parameters 
    x = ti.field(dtype=ti.i32)
    rs = resize_shape // 16
    block1 = ti.root.pointer(ti.ij, (rs, rs))
    block2 = block1.pointer(ti.ij, (4, 4))
    pixel = block2.dense(ti.ij, (4, 4))
    pixel.place(x)

    # Input an image and generate edge detection function
    src = cv2.imread('./pictures/example1.png', 0)
    height, weight = resize_shape, resize_shape
    img = cv2.resize(src, (weight, height))
    [Iy, Ix] = np.gradient(img)
    gValue = 1/ (1 + Ix ** 2 + Iy ** 2)

    # initialize level set function
    c0 = 2
    offset = 8
    phi = c0 * np.ones((height, weight))
    phi[offset: height-offset, offset: weight-offset] = -c0

    # plt.ion()
    # fig = plt.figure()

    time1 = time.time()
    time_init = time1 - start
    print("Initialization Time: ", np.round(time_init, 3), "s")
    print("Taichi Setup: True\n")

    times = []
    for it in tqdm(range(iter_num)):
        it_start = time.time() 
        ## each iteration
        # building activation condition
        [phi_y, phi_x] = np.gradient(phi)
        
        
        # activate sparse field x
        activate(phi_x, phi_y)

        # according to activation field x, measure delta phi
        init_phi(phi)
        dirac_phi = np.zeros((height, weight))
        s = np.zeros((height, weight))
        dps = np.zeros((height, weight))
        dist = np.zeros((height, weight))
        k_x = np.zeros((height, weight))
        k_y = np.zeros((height, weight))
        kxx = np.zeros((height, weight))
        kyy = np.zeros((height, weight))
        phi_laplace = np.zeros((height, weight))
        vx = np.zeros((height, weight))
        vy = np.zeros((height, weight))
        n_x = np.zeros((height, weight))
        n_y = np.zeros((height, weight))
        nxx = np.zeros((height, weight))
        nyy = np.zeros((height, weight))
        area = np.zeros((height, weight))
        edge = np.zeros((height, weight))
        dphi = np.zeros((height, weight))
        
        dirac_phi_taichi(phi, dirac_phi)
        dps_taichi(s, phi_x, phi_y, dps)
        kx_ky_taichi(phi_x, phi_y, dps, k_x, k_y)
        kxx_kyy_taichi(k_x, k_y, kxx, kyy)
        laplace(phi, phi_laplace)
        dist_taichi(mu, kxx, kyy, phi_laplace, dist)
        
        vx_vy_taichi(gValue, vx, vy)
        normalize_phi_taichi(phi_x, s, n_x)
        normalize_phi_taichi(phi_y, s, n_y)
        nxx_nyy_taichi(n_x, n_y, nxx, nyy)
        edge_taichi(lmda, dirac_phi, gValue, vx, vy, n_x, n_y, nxx, nyy, edge)

        area_taichi(alfa0, dirac_phi, gValue, area)
     
        dphi_taichi(area, dist, edge, dphi)

        # update phi with dphi
        update_phi(phi, dphi)

        # deactivate x, reactive x according to new phi in next iteration.
        block1.deactivate_all()

        it_end = time.time()
        times.append(np.round(it_end - it_start, 3))
    times = np.array(times)
    print("Average elapsed time per iteration: ", np.round(np.mean(times), 3), "s")

    end = time.time()
    print("Time Consuming = ", np.round(end - start, 3), "s")
        
        # if it % 50 == 0:
        #     print("show fig for ", it, " iteration")
'''
            fig.clf()
            contours = measure.find_contours(phi, 0)
            ax1 = fig.add_subplot(121)
            ax1.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
            for n, contour in enumerate(contours):
                ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
           
            ax2 = fig.add_subplot(122, projection='3d')
            y, x = phi.shape
            x = np.arange(0, x, 1)
            y = np.arange(0, y, 1)
            X, Y = np.meshgrid(x, y)
            ax2.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
            ax2.contour(X, Y, phi, 0, colors='g', linewidths=2)
            plt.pause(0.3)
'''
