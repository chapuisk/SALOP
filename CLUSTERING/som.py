import numpy as np
from sklearn import preprocessing
from sklearn_som.som import SOM
from utils import load_data, spider_plot, spider_plot_comparison, cluster_projection_plot, cluster_3d_plot, save_boxes, \
    save_clusters


def clustering_with_som(data, nrow=10, ncol=10, sigma=0.5, epochs=100):
    """
    This function clusters the input data using a SOM.

    :param data: The data to cluster
    :param nrow: Number of rows of the SOM
    :param ncol: Number of columns of the SOM
    :param sigma: Learning rate of the SOM
    :param epochs: Number of training epochs of the SOM
    :return: A list containing the label of the data in the same order
    """
    print("Training..")
    p = data.shape[1]
    som = SOM(m=nrow, n=ncol, dim=p, sigma=sigma)
    som.fit(data, epochs=epochs, shuffle=True)
    print("Done.\n")

    print("Clustering..")
    labels = som.predict(data)
    classes = np.unique(labels)
    for i in range(len(classes)):
        labels[labels == classes[i]] = i

    print("number of clusters :", len(np.unique(labels)))
    print("Done.\n")

    return labels


if __name__ == '__main__':
    # LOADS DATA
    X, headers_x, Y, headers_y = load_data(
        path='./data/COMOKIT_Final_Results.csv',
        input_names=["density_ref_contact",
                     "init_all_ages_successful_contact_rate_human",
                     "init_all_ages_factor_contact_rate_asymptomatic",
                     "init_all_ages_proportion_asymptomatic",
                     "init_all_ages_proportion_hospitalisation",
                     "init_all_ages_proportion_icu",
                     "init_all_ages_proportion_dead_symptomatic"],
        output_names=["nb_dead",
                      # "nb_recovered",
                      "step_max_peak",
                      "step_end_epidemiology"]
    )

    # STANDARDIZATION
    X_std = preprocessing.scale(X)
    Y_std = preprocessing.scale(Y)

    # TRAINS SOM AND CLUSTERS
    # Nb of neurons in som = nb of max classes
    labels = clustering_with_som(data=Y_std, nrow=1, ncol=7, sigma=0.5, epochs=1000)

    # VISUALIZATION
    cluster_projection_plot(Y_std, labels, headers_y, "./results/SOM/clusters.png")
    cluster_3d_plot(Y_std[:, 0], Y_std[:, 1], Y_std[:, 2], labels, headers_y, "./results/SOM/3d_clusters.png")
    spider_plot(X_std, labels, headers_x, "./results/SOM/classes_mean_quantile.png")
    spider_plot_comparison(X_std, labels, headers_x, "./results/SOM/classes_comparison.png")
    save_clusters('./data/COMOKIT_Final_Results.csv', labels, "./results/SOM/clustered_data.csv")
    save_boxes(X, labels, headers_x, "./results/SOM/boxes_report.txt")
