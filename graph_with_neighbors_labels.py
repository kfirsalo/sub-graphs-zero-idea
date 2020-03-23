import networkx as nx
import numpy as np
from scipy.special import softmax
# http://networkrepository.com/BA-2-24-60-L2.php
import collections
np.random.seed(0)
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

def data_preparation(nodes_labels):
    all_labels = {x: y-1 for (x, y) in nodes_labels}
    all_nodes1 = list(all_labels.keys())
    all_nodes = []
    for i in range(len(all_nodes1)):
        all_nodes.append(str(all_nodes1[i]))
    return all_nodes


def get_initial_labels_nodes(G, key):
    """
    function that gets the graph and return the nodes that we would like them to be
    in the initial projection
    """
    # a dictionary of the nodes and their degrees
    dict_degrees = dict(G.degree(G.nodes()))
    # a dictionary of the nodes and the average degrees
    dict_avg_neighbor_deg = nx.average_neighbor_degree(G)
    # sort the dictionary
    sort_degrees = sorted(dict_degrees.items(), key=lambda pw: (pw[1], pw[0]))  # list
    # sort the dictionary
    sort_avg_n_d = sorted(dict_avg_neighbor_deg.items(), key=lambda pw: (pw[1], pw[0]))  # list
    # choose only some percents of the nodes with the maximum degree
    top_deg = sort_degrees[int(key * len(sort_degrees)):len(sort_degrees)]
    # choose only some percents of the nodes with the maximum average degree
    top_avgn_deg = sort_avg_n_d[int(key * len(sort_avg_n_d)):len(sort_avg_n_d)]
    # a code to choose the nodes that have maximum degree and also maximum average degree
    tmp_deg = top_deg
    tmp_n_deg = top_avgn_deg
    for i in range(len(top_deg)):
        tmp_deg[i] = list(tmp_deg[i])
        tmp_deg[i][1] = 5
    for i in range(len(top_avgn_deg)):
        tmp_n_deg[i] = list(tmp_n_deg[i])
        tmp_n_deg[i][1] = 10
    # the nodes with the maximal degree- the nodes we want to do the projection on
    final_nodes = np.intersect1d(tmp_n_deg, tmp_deg)
    list_final_nodes = list(final_nodes)
    for i in range(len(list_final_nodes)):
        list_final_nodes[i] = str(list_final_nodes[i])
    return list_final_nodes


def create_dict_neighbors(G):
    """
    Create a dictionary of neighbors.
    :param G: Our graph
    :return: neighbors_dict when value==node and key==set_of_neighbors
    """
    G_nodes = list(G.nodes())
    neighbors_dict = {}
    for i in range(len(G_nodes)):
        node = G_nodes[i]
        neighbors_dict.update({node: set(G[node])})
    return neighbors_dict


def create_dicts_same_nodes(graph_nodes, neighbors_dict, list_of_nodes):
    """
    A function to create useful dictionaries to represent connection between nodes that have the same type, i.e between
    nodes that are in the projection and between nodes that aren't in the projection. It depends on the input.
    :param my_set: Set of the nodes that aren't in the projection OR Set of the nodes that are in the projection
    :param neighbors_dict: Dictionary of all nodes and neighbors (both incoming and outgoing)
    :param node: Current node we're dealing with
    :param dict_out: explained below
    :param dict_in: explained below
    :return: There are 4 possibilities (2 versions, 2 to every version):
            A) 1. dict_node_node_out: key == nodes not in projection , value == set of outgoing nodes not in projection
                 (i.e there is a directed edge (i,j) when i is the key node and j isn't in the projection)
               2. dict_node_node_in: key == nodes not in projection , value == set of incoming nodes not in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j isn't in the projection)
            B) 1. dict_enode_enode_out: key == nodes in projection , value == set of outgoing nodes in projection
                 (i.e there is a directed edge (i,j) when i is the key node and j is in the projection)
               2. dict_enode_enode_in: key == nodes in projection , value == set of incoming nodes in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the projection)
    """
    dict_out = {}
    dict_in = {}
    for node in list_of_nodes:
        if neighbors_dict.get(node) is not None:
            set1 = neighbors_dict[node].intersection(set(graph_nodes))
            if (len(set1)) > 0:
                dict_out.update({node: set1})
                neigh = list(set1)
                for j in range(len(neigh)):
                    if dict_in.get(neigh[j]) is None:
                        dict_in.update({neigh[j]: set([node])})
                    else:
                        dict_in[neigh[j]].update(set([node]))
    return dict_out, dict_in


def make_labels(all_nodes, label_nodes, dict_out, dict_in, p_in, p_out):
    dict_label_nodes = {}
    for i in range(len(label_nodes)):
        label = np.random.randint(0, 2)
        dict_label_nodes.update({label_nodes[i]: label})
    non_label_nodes = list(set(all_nodes) - set(list(dict_label_nodes.keys())))
    cond = len(non_label_nodes)
    itter = 0
    while cond > 0:
        cond = 0
        non_label_nodes = list(set(all_nodes)-set(list(dict_label_nodes.keys())))
        tmp_non_label_nodes = non_label_nodes.copy()
        for i in range(len(tmp_non_label_nodes)):
            current_node = tmp_non_label_nodes[i]
            dict_label_nodes, cond = spread_label_to_neighbors(current_node, dict_out, dict_label_nodes,
                                                              p_out, cond)
            dict_label_nodes, cond = spread_label_to_neighbors(current_node, dict_in, dict_label_nodes,
                                                              p_in, cond)
        print(cond)
        itter += 1
    return dict_label_nodes, itter


def spread_label_to_neighbors(current_node, dict_direction, dict_label_nodes, p, count):
    if dict_direction.get(current_node) is not None:
        neighbors = list(dict_direction[current_node])
        bk = 0
        for i in range(len(neighbors)):
            current_neighbor = neighbors[i]
            if dict_label_nodes.get(current_neighbor) is not None and bk != 1:
                rand = np.random.uniform(0, 1)
                if rand < p:
                    label = dict_label_nodes[current_neighbor]
                    dict_label_nodes.update({current_node: label})
                    count += 1
                    bk = 1
    return dict_label_nodes, count


def plot_iter_prob(p_values, graph_nodes, nodes_for_label, dict_out, dict_in):
    iterations = []
    for p in p_values:
        dict_labels, itter = make_labels(graph_nodes, nodes_for_label, dict_out, dict_in, p, p)
        iterations.append(itter)
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['axes.labelsize'] = 16
    p_axis = p_values
    plt.figure(1)
    plt.title('Number Of Iteration Per Probability')
    plt.xlabel("Probability")
    plt.ylabel("Number Of iterations")
    plt.plot(p_axis, iterations, '-ok', color='blue')
    plt.show()


def main():
    G = nx.read_edgelist("edges.txt")
    graph_nodes = G.nodes()
    # nodes_with_labels = np.loadtxt("node_labels.txt", dtype=int)
    nodes_for_label = get_initial_labels_nodes(G, 0.80) # 0.8 - 122 nodes
    neighbors_dict = create_dict_neighbors(G)
    dict_out, dict_in = create_dicts_same_nodes(graph_nodes, neighbors_dict, graph_nodes)
    dict_labels, itter = make_labels(graph_nodes, nodes_for_label, dict_out, dict_in, 0.5, 0.5)
    print(len(dict_labels))
    return dict_labels

main()
# if __name__ == '__main__':
#     G = nx.read_edgelist("edges.txt")
#     graph_nodes = G.nodes()
#     # nodes_with_labels = np.loadtxt("node_labels.txt", dtype=int)
#     nodes_for_label = get_initial_labels_nodes(G, 0.80) # 0.8 - 122 nodes
#     neighbors_dict = create_dict_neighbors(G)
#     dict_out, dict_in = create_dicts_same_nodes(graph_nodes, neighbors_dict, graph_nodes)
#     dict_labels = make_labels(graph_nodes, nodes_for_label, dict_out, dict_in, 0.001, 0.001)
#     print(len(dict_labels))