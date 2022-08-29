import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils import load_data, spider_plot_comparison, spider_plot, cluster_projection_plot, save_clusters, cluster_3d_plot, \
    cluster_2d_plot


def _best_k(X, maxClasses=10):
    """
    This function finds the best k value to train a kmeans model using the BIC indicator.

    :param X: the data on which to train kmeans
    :param maxClasses: the maximum number of classes for the model (maximum value for k)
    :return: the best k value
    """
    best_score = sys.float_info.max
    best_k = -1
    for k in range(1, maxClasses):
        gmm = GaussianMixture(n_components=k, init_params='kmeans', tol=1e-6, max_iter=10000)
        gmm.fit(X)
        score = gmm.bic(X)
        if score < best_score:
            best_score = score
            best_k = k
    return best_k


def clustering_with_kmeans(X, maxClasses=20):
    """
    This function uses kmeans algorithms to cluster the input data

    :param X: The data to cluster
    :param maxClasses: The desired maximum number of clusters
    :return: A list containing the label of the data in the same order
    """
    print("Training..")
    nbClusters = _best_k(X, maxClasses=maxClasses)
    kmeans = KMeans(n_clusters=nbClusters, random_state=0)
    kmeans.fit(X)
    print("Number of clusters :", nbClusters)
    print("Done.\n")
    return kmeans.labels_


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
                          # "nb_recovered",
                          "step_max_peak",
                          "step_end_epidemiology"]
        )

    labels = clustering_with_kmeans(Y, maxClasses=20)
    cluster_projection_plot(Y, labels, headers_y, "./results/kmeans/clusters.png")
    # cluster_2d_plot(Y, labels, headers_y, "./results/kmeans/2d_clusters.png")
    # cluster_3d_plot(Y, labels, headers_y, "./results/kmeans/3d_clusters.png")
    spider_plot(X, labels, headers_x, "./results/kmeans/classes_mean_quantile.png")
    spider_plot_comparison(X, labels, headers_x, "./results/kmeans/classes_comparison.png")
    # save_clusters('./data/COMOKIT_Final_Results.csv', labels, "./results/kmeans/clustered_data.csv")

