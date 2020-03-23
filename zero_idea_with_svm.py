import networkx as nx
import numpy as np
from scipy.special import softmax
# http://networkrepository.com/BA-2-24-60-L2.php
import collections
import numpy as np
from sklearn import linear_model



def data_preparation(nodes_labels):
    all_labels = {x: y-1 for (x, y) in nodes_labels}
    all_nodes1 = list(all_labels.keys())
    all_nodes = []
    for i in range(len(all_nodes1)):
        all_nodes.append(str(all_nodes1[i]))
    new_ind = np.arange(len(all_nodes1))
    np.random.seed(0)
    np.random.shuffle(new_ind)
    dict_train_labels = {}
    nodes_train = []
    for i in range(int(0.2*len(new_ind))):
        ind = new_ind[i]
        y = all_labels[all_nodes1[ind]]
        dict_train_labels.update({all_nodes[ind]: y})
        nodes_train.append(all_nodes[ind])
    non_label_nodes = list(set(all_nodes)-set(nodes_train))
    # all_nodes = ','.join(str(e) for e in all_nodes1)
    return dict_train_labels, nodes_train, non_label_nodes, all_nodes, all_labels





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


def create_dicts_of_connections(graph_nodes, neighbors_dict, dict_out, dict_in):
    """
     A function that creates 6 dictionaries of connections between different types of nodes.
    :param set_proj_nodes: Set of the nodes that are in the projection
    :param set_no_proj_nodes: Set of the nodes that aren't in the projection
    :param neighbors_dict:
    :return:
    """
    dict_out_out = {}
    final_dict_out_out = {}
    dict_out_in = {}
    final_dict_out_in = {}
    dict_in_out = {}
    final_dict_in_out = {}
    dict_in_in = {}
    final_dict_in_in = {}
    list_out = list(dict_out.keys())
    list_in = list(dict_in.keys())
    # for i in range(len(list_out)):
    #     node = list_out[i]
    list_out_out = {}
    list_out_in = {}
    list_in_out = {}
    list_in_in = {}
    count = 0
    if len(list_out) > 0:
        for key in list_out:
            dict_out_out, dict_out_in = create_dicts_same_nodes(graph_nodes, neighbors_dict, list(dict_out[key]))
            second_order_neighbors_out_out = list(dict_out_out.values())
            second_order_neighbors_out_in = list(dict_out_in.values())
            convert_out_out = set([])
            convert_out_in = set([])
            for item in second_order_neighbors_out_out:
                convert_out_out.update(item)
            for item in second_order_neighbors_out_in:
                convert_out_in.update(item)
            final_dict_out_out.update({key: convert_out_out})
            final_dict_out_out.update({key: convert_out_in})
            count += 1
            if count % 100 == 0:
                print(count)
            # list_dict_out_out = list(dict_out_out.keys())
            # list_dict_out_in = list(dict_out_in.keys())
            # if len(list_dict_out_out) > 0:
            #     for source in list_dict_out_out:
            #         list_out_out.update(dict_out_out[source])
            # if len(list_dict_out_in) > 0:
            #     for source in list_dict_out_in:
            #         list_out_in.update(dict_out_in[source])
            # final_dict_out_out.update({key: set(dist_out_out)})
    if len(list_in) > 0:
        for key in list_in:
            dict_in_out, dict_in_in = create_dicts_same_nodes(graph_nodes, neighbors_dict, list(dict_in[key]))
            second_order_neighbors_in_out = list(dict_in_in.values())
            second_order_neighbors_in_in = list(dict_in_in.values())
            convert_in_out = set([])
            convert_in_in = set([])
            for item in second_order_neighbors_in_out:
                convert_in_out.update(item)
            for item in second_order_neighbors_in_in:
                convert_in_in.update(item)
            final_dict_out_out.update({key: convert_in_out})
            final_dict_out_out.update({key: convert_in_in})
            # list_dict_in_out = list(dict_in_out.keys())
            # list_dict_out_in = list(dict_out_in.keys())
            # if len(list_dict_in_out) > 0:
            #     for source in list_dict_in_out:
            #         list_out_out.update(dict_in_out[source])
            # if len(list_dict_out_in) > 0:
            #     for source in list_dict_out_in:
            #         list_out_in.update(dict_out_in[source])
            # final_dict_out_out.update({key: set(list_out_out)})
            # list_dict_in_out = list(dict_in_out.values())
            # list_dict_in_in = list(dict_in_in.values())
            # if len(list_dict_in_out) > 0:
            #     for j in range(len(list_dict_in_out)):
            #         list_out_out.append(list_dict_in_out[j])
            # if len(list_dict_in_in) > 0:
            #     for j in range(len(list_dict_in_in)):
            #         list_out_out.append(list_dict_in_in[j])
    # for i in range(len(list_in)):
    #     node = list_in[i]
    # dict_in_out, dict_in_in = create_dicts_same_nodes(graph_nodes, neighbors_dict, list_in)
    return final_dict_out_out, final_dict_out_in, final_dict_in_out, final_dict_in_in


def params(non_labels_nodes, dict_out, dict_in, dict_out_out, dict_out_in, dict_in_out, dict_in_in, num_labels):
    [alpha_out, alpha_in, alpha_out_out, alpha_out_in, alpha_in_out, alpha_in_in] = [1, 1, 1, 1, 1, 1]
    score = np.zeros(num_labels)
    params = np.zeros(6)
    # dict_classes = {}
    # for i in range(len(list_of_classes)):
    #     dict_classes.update({list_of_classes[i]: 0})
    dict_node_params = {}
    dict_node_label = {}
    for node in non_labels_nodes:
        if dict_out.get(node) is not None:
            out_neighbors = list(dict_out[node])
            for i in range(len(out_neighbors)):
                if dict_train_labels.get(out_neighbors[i]) is not None:
                    label = dict_train_labels[out_neighbors[i]]
                    score[label] += alpha_out
                    params[0] += alpha_out
        if dict_out_out.get(node) is not None:
            out_out_neighbors = list(dict_out_out[node])
            for i in range(len(out_out_neighbors)):
                if dict_train_labels.get(out_out_neighbors[i]) is not None:
                    label = dict_train_labels[out_out_neighbors[i]]
                    score[label] += alpha_out_out
                    params[1] += alpha_out_out
        if dict_out_in.get(node) is not None:
            out_in_neighbors = list(dict_out_in[node])
            for i in range(len(out_in_neighbors)):
                if dict_train_labels.get(out_in_neighbors[i]) is not None:
                    label = dict_train_labels[out_in_neighbors[i]]
                    score[label] += alpha_out_in
                    params[2] += alpha_out_in
        if dict_in.get(node) is not None:
            in_neighbors = list(dict_in[node])
            for i in range(len(in_neighbors)):
                if dict_train_labels.get(in_neighbors[i]) is not None:
                    label = dict_train_labels[in_neighbors[i]]
                    score[label] += alpha_in
                    params[3] += alpha_in
        if dict_in_out.get(node) is not None:
            in_out_neighbors = list(dict_in_out[node])
            for i in range(len(in_out_neighbors)):
                if dict_train_labels.get(in_out_neighbors[i]) is not None:
                    label = dict_train_labels[in_out_neighbors[i]]
                    score[label] += alpha_in_out
                    params[4] += alpha_in_out
        if dict_in_in.get(node) is not None:
            in_in_neighbors = list(dict_in_in[node])
            for i in range(len(in_in_neighbors)):
                if dict_train_labels.get(in_in_neighbors[i]) is not None:
                    label = dict_train_labels[in_in_neighbors[i]]
                    score[label] += alpha_in_in
                    params[5] += alpha_in_in
        # final_score = softmax(score)
        # label = np.argmax(final_score)
        # dict_node_label.update({node: label})
        dict_node_params.update({node: params})
    return dict_node_params


def svm(dict_features, dict_labels):
    X=[]
    Y=[]
    for i in range(len(nodes_train)):
        X.append(dict_features[nodes_train[i]])
        Y.append(dict_labels[nodes_train[i]])
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X, Y)
    return clf


def zero_idea(model, dict_features):
    non_labeled = list(dict_features.keys())
    dict_predict_non_labels = {}
    for i in range(len(non_labeled)):
        predict = model.predict([dict_features[non_labeled[i]]])
        dict_predict_non_labels.update({non_labeled[i]: predict})
    return dict_predict_non_labels


if __name__ == '__main__':
    G = nx.read_edgelist("edges.txt")
    nodes_with_labels = np.loadtxt("node_labels.txt", dtype=int)
    dict_train_labels, nodes_train, non_label_nodes, all_nodes, all_labels = data_preparation(nodes_with_labels)
    # H = G.to_undirected()
    neighbors_dict = create_dict_neighbors(G)
    dict_out_non_label, dict_in_non_label = create_dicts_same_nodes(all_nodes, neighbors_dict, non_label_nodes)
    dict_out_out, dict_out_in, dict_in_out, dict_in_in = create_dicts_of_connections(nodes_train, neighbors_dict,
                                                                                     dict_out_non_label, dict_in_non_label)
    dict_out_with_label, dict_in_with_label = create_dicts_same_nodes(all_nodes, neighbors_dict, nodes_train)
    dict_out_out_labels, dict_out_in_labels, dict_in_out_labels, dict_in_in_labels = create_dicts_of_connections(nodes_train, neighbors_dict,
                                                                                     dict_out_with_label, dict_in_with_label)
    train_features = params(nodes_train, dict_out_with_label, dict_in_with_label, dict_out_out_labels, dict_out_in_labels,
                         dict_in_out_labels, dict_in_in_labels, 2)
    clf = svm(train_features, dict_train_labels)
    non_label_nodes_features = params(non_label_nodes, dict_out_non_label, dict_in_non_label, dict_out_out, dict_out_in, dict_in_out,
                                    dict_in_in, 2)
    dict_node_pre_label = zero_idea(clf, non_label_nodes_features)
    # sort_dict_node_pre_label = dict(collections.OrderedDict(sorted(dict_node_pre_label.items())))
    # sort_all_labels = dict(collections.OrderedDict(sorted(all_labels.items())))
    # keys = list(all.keys())
    count = 0
    for i in range(len(non_label_nodes)):
        if dict_node_pre_label.get(non_label_nodes[i]) is not None:
            if all_labels[int(non_label_nodes[i])] == dict_node_pre_label[non_label_nodes[i]]:
                count += 1
    acc = 100*count/float(len(dict_node_pre_label))
    print("acc", acc)




