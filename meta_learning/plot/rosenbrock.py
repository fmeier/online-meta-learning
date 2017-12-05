
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter


def plot_func(dir_path_plot,
              plot_name,
              min_val,
              max_val,
              steps,
              x1_path=None,
              x2_path=None):
    plt.clf()
    fig = plt.figure(figsize=(20, 10))
    plt.title(plot_name)
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.linspace(min_val, max_val, steps)
    Y = np.linspace(min_val, max_val, steps)
    X, Y = np.meshgrid(X, Y)
    # Rosenbrock function
    # (p.a - x[1])^2 + p.b * (x[2] - x[1]^2)^2;
    a = 1.0
    b = 100.0
    Z = (a - X)**2 + b * (Y - X**2)**2

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    z_path = None
    if x1_path is not None and x2_path is not None:
        z_path = (a - x1_path)**2 + b * (x2_path - x1_path**2)**2
        ax.plot(x1_path, x2_path, z_path, color='k', linewidth=2.0)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 180)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=4)
    plot_path = os.path.join(dir_path_plot, plot_name + '_func')
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_path, dpi=100)
    plt.show()
