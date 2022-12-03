
import cv2
import time
from tqdm import tqdm
import numpy as np
import taichi as ti
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

resize_shape = 1024
img2d = ti.types.ndarray()

# Calculate delta phi, return a numpy array
def get_dphi(img, phi, gValue, phi_x, phi_y):

    # Neumann condition
    phi[np.ix_([0, -1], [0, -1])] = phi[np.ix_([2, -3], [2, -3])]
    phi[np.ix_([0, -1]), 1:-1] = phi[np.ix_([2, -3]), 1:-1]
    phi[1:-1, np.ix_([0, -1])] = phi[1:-1, np.ix_([2, -3])]

    dfunc = (1 / 2 / epsilon) * (1 + np.cos(np.pi * phi / epsilon))
    conds = (phi <= epsilon) & (phi >= -epsilon)
    dirac_phi = dfunc * conds
    
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
   
    return dist_reg_term + edge_term + area_term


if __name__ == "__main__":
    
    start = time.time()

    # Initialize parameters 
    mu = 0.2
    timestep = 0.2 / mu
    iter_num = 400
    lmda = 1.5
    alfa0 = 1.5
    epsilon = 1.5
    sigma = 1.5

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

    #plt.ion()
    #fig = plt.figure()

    print("Initialization Time: ", np.round(time.time()-start, 3), "s")
    print("Taichi Setup: False")

    times = []

    for it in tqdm(range(iter_num)):
        it_start = time.time()
        time_it = time.time()

        ## each iteration
        # building activation condition
        [phi_y, phi_x] = np.gradient(phi)
        
        # according to activation field x, measure delta phi
        dphi = get_dphi(img, phi, gValue, phi_x, phi_y)

        # update phi with dphi
        phi += dphi
        
        it_end = time.time()
        
        times.append(it_end - it_start)

    times = np.array(times)
    print("Average elapsed time per iteration:", np.round(np.mean(times), 3),"s")

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
