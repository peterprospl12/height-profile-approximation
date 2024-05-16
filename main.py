import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lagrange_cython

def plot_height_profile(path, header=True):
    # Wczytanie danych
    if header:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, header=None)

    # Plot pierwotnych danych
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], color='blue')
    plt.fill_between(data.iloc[:, 0], 0, data.iloc[:, 1], color='skyblue')
    plt.fill_between(data.iloc[:, 0], data.iloc[:, 1], max(data.iloc[:, 1]), color='darkgoldenrod')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Height Profile of {os.path.splitext(os.path.basename(path))[0]}')

    # Interpolacja danych
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    x_new = chebyshev_nodes(min(x), max(x), len(x))
    y_new = lagrange_cython.lagrange_interpolation_cython(x, y, x_new)

    # Plot interpolowanych danych
    plt.subplot(1, 2, 2)
    plt.plot(x_new, y_new, label='Interpolated', color='green')
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
    plt.scatter(x, y, color='red')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def lagrange_interpolation(x, y, x_new):
    n = len(x)
    m = len(x_new)
    y_new = np.zeros(m)
    for i in range(m):
        for j in range(n):
            p = 1
            for k in range(n):
                if k != j and (x[j] - x[k]) != 0:  # Add a check for division by zero
                    p *= (x_new[i] - x[k]) / (x[j] - x[k])
            y_new[i] += y[j] * p
    return y_new


def chebyshev_nodes(a, b, n):
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))

def main():
    plot_height_profile('data/MountEverest.csv')



if __name__ == "__main__":
    main()
