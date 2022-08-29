import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from math import ceil, floor
from matplotlib import pyplot as plt
from sklearn import preprocessing


def load_data(path, input_names, output_names, standardize=True):
    """
    This function load data from a .csv file

    :param path: path to the .csv file containing the data
    :param input_names: array containing names of input columns
    :param output_names: array containing names of output columns
    :param standardize: if True apply data standardization
    :return: inputs, input headers, outputs, output headers
    """
    data = pd.read_csv(path)
    X = data[input_names]
    Y = data[output_names]

    headers_x = X.columns[:]
    x = preprocessing.scale(X.values) if standardize else X.values
    headers_y = Y.columns[:]
    y = preprocessing.scale(Y.values) if standardize else Y.values
    return x, headers_x, y, headers_y


def save_clusters(input_path, labels, output_path):
    print("Saving clusters to " + output_path + "..")
    data = pd.read_csv(input_path)
    data["cluster"] = labels
    data.to_csv(output_path)
    print("Saved.\n")


def cmap_builder(n):
    """
    Build a color map
    :param n: Number of desired colors
    :return:
    """
    if n == 2:
        colors = matplotlib.colors.ListedColormap(['green', 'crimson'])
    elif n < 9:
        colors = plt.cm.get_cmap("Set1", n)
    else:
        colors = plt.cm.get_cmap("tab20", n)
    return colors


def cluster_2d_plot(x, y, labels, headers, path):
    """
    Plot a 2d clustering problem
    :param x: x coordinates
    :param y: y coordinates
    :param labels: array containing the cluster labels of the data
    :param headers: headers of the data
    :param path: path to the file to save the plot
    """
    print("Building 2d plot..")
    # Test dim
    if headers.shape[0] != 2:
        print("Headers must be in dimension 2, found :", headers.shape[0])
        return

    # Build colors
    colors = cmap_builder(len(np.unique(labels)))

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=labels, cmap=colors)
    ax.set_xlabel(headers[0])
    ax.set_ylabel(headers[1])
    items = []
    for c in colors.colors:
        items.append(mpatches.Patch(color=c, label=len(items)))
    fig.legend(handles=items)
    fig.savefig(path, transparent=False)
    plt.close(fig)
    print("Done.\n")


def cluster_3d_plot(x, y, z, labels, headers, path):
    """
    Plot a 3d clustering problem
    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :param labels: array containing the cluster labels of the data
    :param headers: headers of the data
    :param path: path to the file to save the plot
    """
    print("Building 3d plot..")
    # Test dim
    if headers.shape[0] != 3:
        print("Headers must be in dimension 3, found :", headers.shape[0])
        return
    # Build Colors
    colors = cmap_builder(len(np.unique(labels)))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=labels, cmap=colors, alpha=0.3)
    ax.set_xlabel(headers[0])
    ax.set_ylabel(headers[1])
    ax.set_zlabel(headers[2])
    items = []
    for c in colors.colors:
        items.append(mpatches.Patch(color=c, label=len(items)))
    fig.legend(handles=items)
    fig.savefig(path, transparent=False)
    plt.close(fig)
    print("Done.\n")


def cluster_projection_plot(data, labels, headers, path):
    """
    This function plots 2D projections of the clusters

    :param data: the data
    :param labels: array containing the cluster labels of the data
    :param headers: headers of the data
    :param path: path to the file to save the plot
    """
    nb_param = data.shape[1]
    small = nb_param * 5
    medium = nb_param * 7
    big = nb_param * 15
    print(path + "    Plotting..")
    fig, axs = plt.subplots(nb_param, nb_param, figsize=(medium, medium), frameon=False)
    fig.suptitle("Clusters", fontsize=big)
    colors = cmap_builder(len(np.unique(labels)))

    for i in range(nb_param):
        for j in range(nb_param):
            if i == j:
                plt.rc('font', size=small)
                axs[i, i].text(0.05, 0.5, headers[i])
                axs[i, i].set_visible = True
                axs[i, i].get_xaxis().set_visible(False)
                axs[i, i].get_yaxis().set_visible(False)
            if i < j:
                px = data[:, j]
                py = data[:, i]
                axs[i, j].scatter(px, py, c=labels, cmap=colors)
                axs[i, j].get_xaxis().set_visible(True)
                axs[i, j].get_yaxis().set_visible(True)
                axs[i, j].set_visible = True
            if i > j:
                axs[i, j].set_visible(False)
                axs[i, j].get_yaxis().set_visible(False)
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)
                axs[i, j].set_visible = False
                axs[i, j].patch.set_facecolor('none')

    items = []
    for c in colors.colors:
        items.append(mpatches.Patch(color=c, label=len(items)))
    fig.legend(handles=items, loc="center right", title="Classes", framealpha=1)
    fig.savefig(path, transparent=False)
    plt.close(fig)
    print("Done.\n")


def spider_plot(data, labels, headers, path):
    """
    This function builds the spider plot of each class with q1 mean and q2 of each parameter

    :param data: the data
    :param labels: array containing the cluster labels of the data
    :param headers: headers of the data
    :param path: path to the file to save the plot
    """
    print(path + "    Plotting..")
    # number of variable
    p = data.shape[1]
    # Clusters
    categories = np.unique(labels)
    # Create a plot per cluster
    shape = ceil(len(categories) ** 0.5)
    fig, axs = plt.subplots(figsize=(shape * 10, shape * 10), nrows=shape, ncols=shape,
                            subplot_kw={'projection': 'polar'})
    # padding
    plt.subplots_adjust(wspace=0.5)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    theta = np.linspace(0, 2 * np.pi, p, endpoint=False)
    theta = np.append(theta, theta[0])
    headers = np.append(headers, headers[0])

    for ax, cat in zip(axs.flat, categories):
        # title
        title = "Class " + str(cat)
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        # labels
        ax.set_xticks(theta)
        ax.set_xticklabels(headers, fontsize=shape * 5)

        # compute q1 mean q3
        q1 = np.quantile(data[labels == cat], 0.25, axis=0)
        q1 = np.append(q1, q1[0])
        mean = np.mean(data[labels == cat], axis=0)
        mean = np.append(mean, mean[0])
        q3 = np.quantile(data[labels == cat], 0.75, axis=0)
        q3 = np.append(q3, q3[0])

        # plot q1 mean and q3
        ax.plot(theta, q1, color="indianred", linestyle='dashed')
        ax.plot(theta, mean, color="red")
        ax.plot(theta, q3, color="indianred", linestyle='dashed')

        ymin = floor(min(np.concatenate([q1, mean, q3])))
        ymax = ceil(max(np.concatenate([q1, mean, q3])))
        ax.set_yticks(np.arange(ymin, ymax, 0.5))
        ax.set_yticklabels(np.arange(ymin, ymax, 0.5), fontsize=shape * 5)
    # save fig
    fig.savefig(path, transparent=False)
    plt.close(fig)
    print("Done.\n")


def spider_plot_comparison(data, labels, headers, path, classes=None):
    """
    This function builds a spider plot the contains the means value of each class for each parameter

    :param data: the data
    :param labels: array containing the cluster labels of the data
    :param headers: headers of the data
    :param path: path to the file to save the plot
    :param classes: label of classes to compare, if none every classes will be plot
    """
    print(path + "    Plotting..")
    # number of variable
    p = data.shape[1]
    # Clusters
    categories = np.unique(labels) if classes is None else classes
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    theta = np.linspace(0, 2 * np.pi, p, endpoint=False)
    theta = np.append(theta, theta[0])
    headers = np.append(headers, headers[0])

    # polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title("Classes comparison")
    # labels
    ax.set_xticks(theta)
    ax.set_xticklabels(headers, fontsize=7)

    # colors
    colors = cmap_builder(len(categories))
    patches = []

    # ymin and ymax for y label range
    ymin = sys.float_info.max
    ymax = sys.float_info.min
    # Plot mean for each class
    for cat in categories:
        # compute mean
        mean = np.mean(data[labels == cat], axis=0)
        mean = np.append(mean, mean[0])

        # update min and max
        local_min = floor(min(mean))
        local_max = ceil(max(mean))
        ymin = local_min if local_min < ymin else ymin
        ymax = local_max if local_max > ymax else ymax

        # get color
        rgb = colors(cat)
        rgba = list(rgb)
        rgba[3] = 0.1
        rgba = tuple(rgba)
        # plot mean
        ax.plot(theta, mean, color=rgb)
        ax.fill(theta, mean, color=rgba)
        patches.append(mpatches.Patch(color=rgb, label=cat))

    # add legend
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1.1), prop={'size': 5})
    # set y label
    ax.set_yticks(np.arange(ymin, ymax, 0.5))
    ax.set_yticklabels(np.arange(ymin, ymax, 0.5), fontsize=5)

    # save fig
    fig.savefig(path, transparent=False)
    plt.close(fig)
    print("Done.\n")
