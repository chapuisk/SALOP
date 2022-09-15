import numpy as np
from sklearn import preprocessing

from utils import load_data, cluster_2d_plot, spider_plot_comparison, spider_plot, save_boxes, save_clusters


def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    print("Computing pareto front..")
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    print("Done.\n")
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


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
                      # "step_max_peak",
                      "step_end_epidemiology"]
    )

    # STANDARDIZATION
    X_std = preprocessing.scale(X)
    Y_std = preprocessing.scale(Y)

    # COMPUTE PARETO FRONT
    is_efficient = is_pareto_efficient(Y_std)

    # VISUALIZATION
    order = np.argsort(is_efficient) # to plot the pareto front in the foreground you need to sort the data
    cluster_2d_plot(Y_std[order, 0], Y_std[order, 1], is_efficient[order], headers_y[:2], "./results/pareto/2d_clusters.png")
    # cluster_projection_plot(Y_std[order], is_efficient[order], headers_y, "./results/pareto/clusters.png")
    # cluster_3d_plot(Y_std[:, 0], Y_std[:, 1], Y_std[:, 2], is_efficient, headers_y, "./results/pareto/3d_clusters.png")
    spider_plot_comparison(X_std, is_efficient, headers_x, "./results/pareto/classes_comparison.png", [1])
    spider_plot(X_std, is_efficient, headers_x, "./results/pareto/classes_mean_quantile.png")
    save_clusters('./data/COMOKIT_Final_Results.csv', is_efficient, "./results/pareto/clustered_data.csv")
    save_boxes(X, is_efficient, headers_x, "./results/pareto/boxes_report.txt", [1])
