import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lagrange_cython


def plot_height_profile(x, y, x_new, y_new, data, path, name):
    plt.figure(figsize=(8, 5))
    plt.plot(x_new, y_new, label=f'Interpolated {name}', color='red')
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
    plt.scatter(x, y, color='green')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'{name} Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lagrange_interpolation(path, nodes, header=True):
    # Load data
    if header:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, header=None)

    # Chebyshev nodes
    indexes = chebyshev_nodes(0, len(data) - 1, nodes).astype(int)
    x = data.iloc[:, 0].values[indexes].tolist()
    y = data.iloc[:, 1].values[indexes].tolist()

    x_new = linspace(min(x), max(x), int(max(x)))
    y_new = lagrange_cython.lagrange_interpolation_cython(x, y, x_new)

    plot_height_profile(x, y, x_new, y_new, data, path, "Lagrange Chebyshev")

    # Linspace Nodes
    indexes = [int(i) for i in linspace(0, len(data) - 1, nodes)]
    x = data.iloc[:, 0].values[indexes].tolist()
    y = data.iloc[:, 1].values[indexes].tolist()
    x_new = linspace(min(x), max(x), int(max(x)))
    y_new = lagrange_cython.lagrange_interpolation_cython(x, y, x_new)

    plot_height_profile(x, y, x_new, y_new, data, path, "Lagrange Linspace")


def spline_interpolation(x, y):
    X = []
    values = []
    size = len(x) - 1
    for i in range(size):
        first_row = [0] * 4 * size
        second_row = [0] * 4 * size

        values.append(y[i])
        values.append(y[i + 1])
        for j in range(4):
            first_row[i * 4 + j] = x[i] ** (3 - j)
            second_row[i * 4 + j] = x[i + 1] ** (3 - j)
        X.append(first_row)
        X.append(second_row)

    for i in range(1, size):
        row = [0] * 4 * size
        index = (i - 1) * 4
        row[index] = 3 * x[i] ** 2
        row[index + 1] = 2 * x[i]
        row[index + 2] = 1
        row[index + 4] = -row[index]
        row[index + 5] = -row[index + 1]
        row[index + 6] = -row[index + 2]
        X.append(row)
        values.append(0)

    for i in range(1, size):
        row = [0] * 4 * size
        index = (i - 1) * 4
        row[index] = 6 * x[i]
        row[index + 1] = 2
        row[index + 4] = -row[index]
        row[index + 5] = -row[index + 1]
        X.append(row)
        values.append(0)

    row = [0] * 4 * size
    row[0] = 6 * x[0]
    row[1] = 2
    X.append(row)

    row = [0] * 4 * size
    row[-4] = 6 * x[-1]
    row[-3] = 2
    X.append(row)
    values.append(0)
    values.append(0)

    solutions = np.linalg.solve(X, values).tolist()
    result = []
    for i in range(size):
        result.append(solutions[i * 4: i * 4 + 4])
    return result


def plot_spline_interpolation(path, nodes, header=True):
    # Load data
    if header:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, header=None)

    indexes = [int(i) for i in linspace(0, len(data) - 1, nodes)]
    x = data.iloc[:, 0].values[indexes].tolist()
    y = data.iloc[:, 1].values[indexes].tolist()

    x_new = linspace(min(x), max(x), int(max(x)))
    solution = spline_interpolation(x, y)
    sol_idx = 0
    y_new = []
    for i in range(len(x_new)):
        if x_new[i] > x[sol_idx + 1] and sol_idx < len(solution) - 1:
            sol_idx += 1
        value = solution[sol_idx][0] * x_new[i] ** 3 + solution[sol_idx][1] * x_new[i] ** 2 + solution[sol_idx][2] * \
                x_new[i] + solution[sol_idx][3]
        y_new.append(value)

    plot_height_profile(x, y, x_new, y_new, data, path, "Cubic Spline")


def chebyshev_nodes(a, b, n):
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))


def linspace(a, b, n):
    offset = (b - a) / (n - 1)
    return [a + i * offset for i in range(n)]


def main():
    for i in range(14, 15):
        plot_lagrange_interpolation(f'data/MountEverest.csv', i)
        plot_spline_interpolation(f'data/MountEverest.csv', i)


if __name__ == "__main__":
    main()
