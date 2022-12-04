import numpy as np

def SurfaceNormal(phi):
    [phi_y, phi_x] = np.gradient(phi)
    print("phi_x = ")
    print(phi_x)
    print("phi_y = ")
    print(phi_y)
    h, w = phi.shape
    normal_field = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            normal_field[i][j] = [-phi_x[i][j], -phi_y[i][j]]

    U = normal_field[:, :, 0]
    V = normal_field[:, :, 1]

    return U, V

if __name__ == "__main__":
    phi = np.random.randint(low=0, high=5, size=(3, 3))
    U, V = SurfaceNormal(phi)
    print("U = ")
    print(U)
    print("V = ")
    print(V)
