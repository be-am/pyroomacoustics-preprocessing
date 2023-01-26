import numpy as np

import matplotlib.pyplot as plt

def cot(x):
    return 1/np.tan(x)


def DoughertyLogSpiral(center, rmax, rmin, v, n_mics, direction = 'vertical'):

    list_coords = []

    l_max = rmin * np.sqrt(1 + cot(v)**2) / (cot(v)) * (rmax/rmin - 1)
    l_n = [i/(n_mics-1) * l_max for i in range(n_mics)] 
    # l_mx = rmin * np.sqrt(1 + cot(v)**2) / (cot(v) * (rmax/rmin - 1))

    Theta = [np.log(1 + cot(v) * x / (rmin*np.sqrt(1 + cot(v)**2)))/cot(v) for x in l_n]

    R = [rmin * np.e**(cot(v)*x) for x in Theta]

    X = [ r * np.cos(theta) for theta, r in zip(Theta, R)]
    Y = [ r * np.sin(theta) for theta, r in zip(Theta, R)]

    
    if direction == 'vertical':
        for x, y, in zip(X, Y):
            list_coords.append([center[0], center[1] + x, center[2] + y])

    elif direction == 'horizontal':
        for x, y, in zip(X, Y):
            list_coords.append([center[0]+ x, center[1]+ y, center[2]])

    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)


def circular_3d_coords(center, radius, num, direction = 'vertical'):

    list_coords = []

    if direction == 'vertical':
        for i in range(num):
            list_coords.append([center[0], center[1] + radius*np.sin(2*i*np.pi/num), center[2] + radius*np.cos(2*i*np.pi/num)])

    elif direction == 'horizontal':
        for i in range(num):
            list_coords.append([center[0]+ radius*np.sin(2*i*np.pi/num), center[1]+ radius*np.cos(2*i*np.pi/num), center[2] ])
    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)


if __name__ == "__main__":

    X, Y = DoughertyLogSpiral(rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = 112)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X, Y, c ="blue")
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.show()