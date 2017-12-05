
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_data_var(dir_path_plot,
                  plot_name,
                  x_name,
                  y_names_plot,
                  y_names_scatter,
                  data):
    fig = plt.figure(figsize=(20, 10))
    plt.title(plot_name)
    ax = fig.add_subplot(111)
    x = data[x_name]
    indices = np.argsort(x)
    x = x[indices]
    for y_name in y_names_plot:
        y_mean = y_name + '_mean'
        if y_mean in data:
            ax.plot(x, data[y_mean][indices], label=y_name)
            y_var = y_name + '_var'
            if y_var in data:
                ax.fill_between(
                    x,
                    data[y_mean][indices] - data[y_var][indices],
                    data[y_mean][indices] + data[y_var][indices])
        else:
            ax.plot(x, data[y_name][indices], label=y_name)

    for y_name in y_names_scatter:
        ax.scatter(x, data[y_name][indices], label=y_name)
    plot_path = os.path.join(dir_path_plot, plot_name)
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_path, dpi=100)
    plt.show()


def plot_value_by_step(dir_path_plot,
                       plot_name,
                       value_name,
                       step,
                       value, logscale=False):
    fig = plt.figure(figsize=(20, 10))
    if logscale:
        plt.yscale('log')
    plt.title(plot_name + '_' + value_name)
    ax = fig.add_subplot(111)
    ax.plot(step, value)
    plot_path = os.path.join(dir_path_plot, plot_name + '_' + value_name)
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_path, dpi=100)
    plt.show()


def plot_values_by_step(dir_path_plot,
                        plot_name,
                        value_name,
                        plots, logscale=False):
    fig = plt.figure(figsize=(20, 10))
    if logscale:
        plt.yscale('log')
    plt.title(plot_name)
    ax = fig.add_subplot(111)
    for name, data in sorted(plots.items()):
        ax.plot(data['step'], data[value_name], label=name)
    plot_path = os.path.join(dir_path_plot, plot_name + '_' + value_name)
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(plot_path, dpi=100)
    plt.show()
