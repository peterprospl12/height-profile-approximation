import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lagrange_cython

nodes_to_plot = [10, 20, 40, 60, 100]


def plot_mse(x, name, mse_lagrange_cheb, mse_lagrange_lin, mse_spline, mse=True):
    plt.figure(figsize=(9, 6))

    if mse:
        ylabel = 'Mean Squared Error'
    else:
        ylabel = 'Differential error'

    plt.plot(x, mse_lagrange_lin, label='Lagrange lin', color='green')
    plt.plot(x, mse_lagrange_cheb, label='Lagrange cheb', color='blue')
    plt.plot(x, mse_spline, label=f'Spline', color='red')
    plt.yscale('log')
    plt.title(f'{name} {ylabel} for different interpolation methods')
    plt.xlabel('Number of nodes')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if mse:
        plt.savefig(f'plots/{name}_mse.png')
    else:
        plt.savefig(f'plots/{name}_msek.png')
    plt.show()


def plot_height_profile(x, y, x_new, y_new, data, path, name, nodes):
    plt.figure(figsize=(7, 9))
    plt.plot(x_new, y_new, label=f'Interpolated {name}', color='red')
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
    plt.scatter(x, y, color='green')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'{name} Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}, {nodes} nodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{os.path.splitext(os.path.basename(path))[0]}_{name}_{nodes}_nodes.png')
    plt.show()


def plot_spline_interpolation(path, nodes, header=True):
    # Load data
    if header:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, sep=" ", header=None)

    indexes = [int(i) for i in linspace(0, len(data) - 2, nodes)]
    x = data.iloc[:, 0].values[indexes].tolist()
    y = data.iloc[:, 1].values[indexes].tolist()

    x_new = linspace(min(x), max(x), len(data) - 1)
    start_time = time.time()
    solution = spline_interpolation(x, y)
    sol_idx = 0
    y_new = []
    for i in range(len(x_new)):
        if x_new[i] > x[sol_idx + 1] and sol_idx < len(solution) - 1:
            sol_idx += 1
        value = solution[sol_idx][0] * x_new[i] ** 3 + solution[sol_idx][1] * x_new[i] ** 2 + solution[sol_idx][2] * \
                x_new[i] + solution[sol_idx][3]
        y_new.append(value)
    print(f"Spline interpolation {nodes} nodes: ", time.time() - start_time)
    if nodes in nodes_to_plot:
        plot_height_profile(x, y, x_new, y_new, data, path, "Cubic Spline", nodes)
    return [[y_new[idx] for idx in indexes], y]


def plot_lagrange_interpolation(path, nodes, header=True):
    # Load data
    if header:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, sep=" ", header=None)

    # Chebyshev nodes
    indexes = [int(i) for i in chebyshev_nodes(0, len(data) - 1, nodes)]
    x = data.iloc[:, 0].values[indexes].tolist()
    y_cheb = data.iloc[:, 1].values[indexes].tolist()

    x_new = linspace(min(x), max(x), len(data) - 1)
    start_time = time.time()
    y_new_cheb = lagrange_cython.lagrange_interpolation_cython(x, y_cheb, x_new)
    print(f"Chebyshev Lagrange interpolation {nodes} nodes: ", time.time() - start_time)

    # plot_height_profile(x, y_cheb, x_new, y_new_cheb, data, path, "Lagrange Chebyshev", nodes)

    # Linspace Nodes
    indexes_lin = [int(i) for i in linspace(0, len(data) - 2, nodes)]
    x = data.iloc[:, 0].values[indexes_lin].tolist()
    y = data.iloc[:, 1].values[indexes_lin].tolist()
    x_new = linspace(min(x), max(x), len(data) - 1)
    start_time = time.time()
    y_new_lin = lagrange_cython.lagrange_interpolation_cython(x, y, x_new)
    print(f"Linsapce Lagrange interpolation {nodes} nodes: ", time.time() - start_time)
    # plot_height_profile(x, y, x_new, y_new_lin, data, path, "Lagrange Linspace", nodes)

    if nodes in nodes_to_plot:
        plt.figure(figsize=(7, 9))
        plt.subplot(2, 1, 1)
        plt.plot(x_new, y_new_lin, label='Interpolated Lagrange Linspace', color='red')
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
        plt.scatter(x, y, color='green')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title(
            f'Lagrange Linspace Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}, {nodes} nodes')
        plt.legend()
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        plt.plot(x_new, y_new_cheb, label='Interpolated Lagrange Chebyshev', color='red')
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original', color='blue')
        plt.scatter(x, y, color='green')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height (m)')
        plt.title(
            f'Lagrange Chebyshev Interpolated Height Profile of {os.path.splitext(os.path.basename(path))[0]}, {nodes} nodes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/{os.path.splitext(os.path.basename(path))[0]}_{nodes}_nodes.png')
        plt.show()

    return [[y_new_cheb[index] for index in indexes],
            y_cheb,
            [y_new_lin[index] for index in indexes_lin],
            y]


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


def chebyshev_nodes(a, b, n):
    return [0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * (i + 1) - 1) * math.pi / (2 * n)) for i in range(n)]


def linspace(a, b, n):
    offset = (b - a) / (n - 1)
    return [a + i * offset for i in range(n)]


def calc_mse(y_new, y_original):
    squared_diff = sum([(y_n - y_o) ** 2 for y_n, y_o in zip(y_new, y_original)])
    return squared_diff / len(y_original)


def main():
    nodes_start = 10
    nodes_end = 101
    mse_lagrange_cheb = []
    mse_lagrange_lin = []
    mse_spline = []

    msek_lagrange_cheb = []
    msek_lagrange_lin = []
    msek_spline = []

    y_new_lagrange_cheb = None
    y_new_lagrange_lin = None
    y_new_spline = None

    path = 'data/SpacerniakGdansk.csv'
    name = os.path.splitext(os.path.basename(path))[0]
    for i in range(nodes_start, nodes_end):
        lagrange = plot_lagrange_interpolation(path, i)
        spline = plot_spline_interpolation(path, i)

        if y_new_lagrange_cheb is not None:
            msek_lagrange_cheb.append(calc_mse(lagrange[0], y_new_lagrange_cheb))
            msek_lagrange_lin.append(calc_mse(lagrange[2], y_new_lagrange_lin))
            msek_spline.append(calc_mse(spline[0], y_new_spline))

        y_new_lagrange_cheb = lagrange[0]
        y_new_lagrange_lin = lagrange[2]
        y_new_spline = spline[0]

        mse_lagrange_cheb.append(calc_mse(lagrange[0], lagrange[1]))
        mse_lagrange_lin.append(calc_mse(lagrange[2], lagrange[3]))
        mse_spline.append(calc_mse(spline[0], spline[1]))

    x = list(range(nodes_start, nodes_end))
    plot_mse(x, name, mse_lagrange_cheb, mse_lagrange_lin, mse_spline)
    x = list(range(nodes_start, nodes_end - 1))
    plot_mse(x, name, msek_lagrange_cheb, msek_lagrange_lin, msek_spline, mse=False)


if __name__ == "__main__":
    main()
