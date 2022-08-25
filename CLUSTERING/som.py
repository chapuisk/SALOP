import numpy as np
from sklearn_som.som import SOM
from utils import load_data, spider_plot, spider_plot_comparison, cluster_plot


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
                      "nb_recovered",
                      "step_max_peak",
                      "step_end_epidemiology"]
    )
    labels = clustering_with_som(data=Y, nrow=1, ncol=7, sigma=0.5, epochs=1000)
    cluster_plot(Y, labels, headers_y, "./results/SOM/clusters.png")
    spider_plot(X, labels, headers_x, "./results/SOM/classes_mean_quantile.png")
    spider_plot_comparison(X, labels, headers_x, "./results/SOM/classes_comparison.png")
