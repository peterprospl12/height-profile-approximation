import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lagrange_cython


def plot_height_profile(path, nodes,header=True):
    # Load data
    if header:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, header=None)

    # Plot original data
    plt.figure(figsize=(20, 18))

    plt.subplot(3, 1, 1)
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], color='blue')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Height Profile of {os.path.splitext(os.path.basename(path))[0]}')


    # Chebyshev nodes
    indexes = chebyshev_nodes(0, len(data) - 1, nodes).astype(int)
    x = data.iloc[:, 0].values[indexes]
    y = data.iloc[:, 1].values[indexes]

    x_new = chebyshev_nodes(min(x), max(x), int(max(x)))
    y_new = lagrange_cython.lagrange_interpolation_cython(x, y, x_new)


    # Plot chebyshev nodes
    plt.subplot(3, 1, 2)
    plt.plot(x_new, y_new, label='Interpolated Chebyshev', color='red')
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
    plt.scatter(x, y, color='red')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}')
    plt.legend()

    plt.tight_layout()

    # Linspace Nodes
    indexes = np.linspace(0, len(data) - 1, nodes).astype(int)
    x = data.iloc[:, 0].values[indexes]
    y = data.iloc[:, 1].values[indexes]
    x_new = np.linspace(min(x), max(x), int(max(x)))
    y_new = lagrange_cython.lagrange_interpolation_cython(x, y, x_new)

    # PLots linspace
    plt.subplot(3, 1, 3)
    plt.plot(x_new, y_new, label='Interpolated Linspace', color='red')
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
    plt.scatter(x, y, color='red')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def chebyshev_nodes(a, b, n):
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))



def main():
    for i in range(1, 10):
        plot_height_profile(f'data/SpacerniakGdansk.csv', 10*i)


if __name__ == "__main__":
    main()
