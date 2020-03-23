# sub-graphs
zero idea for learn on 2 order neighbors sub graphs.
The problem: assume we have a graph G some of his vertices with labels and we want to predict for each vertex his label. G has many vertices and edges so we can’t learn on the whole graph. we want for every vertex look on a sub graph which the vertex makes: his one order and two order neighbors.
our mission: be v a node in our graph G and given his one and two order neighbors sub graph. we want to predict the label of vertex v based on his sub graph.

# Zero idea
what is the zero idea?
let v be a vertex. 
xi1 his out neighbor (if it has one).
xi11 out neighbor of his out neighbor.
xi12 in neighbor of his out neighbor.
xi2 his in neighbor.
xi21 out neighbor of his in neighbor.
xi12 in neighbor of his in neighbor.

the zero idea is sort of veirsion of knn.
for every class we sum the weighted count of vertices in the sub graph (that in that class):
ai = \sum_{j=1}^{6}\sum_{n=1}^{k}wnj
j run on the types of neighbors v have. k is the number of neighbors we know in class i. the weight w is based on the importance of every neighbor(usually out neighbors more important than in neighbors and one order neighbor more important than two order neighbor).
y=argmax(softmax(ai))
the class with the biggest count is our prediction y to the vertex v.
I test that idea on a graph with 10636 vertices and 639,750 edges.
In this graph every vertex has a label from two classes. I split the labels into 20% train and 80% test. the result was almost random (50.25% accuracy). 

# zero idea with svm
I thought maybe the weights I chose can be more accurate, so I run an svm algorithm on my train vertices to estimate the best parameters. the result was still bad (50.36% accuracy).

# simulation: add labels based on neighbors
I build a simulation to check how the zero idea behave when every vertice get his out neighbor label in p_out and in neighbor in p_in.
so I started with give 122 vertices with the higher rank and higher neighbors rank an initial random label. for every vertex v that hasn’t a label, if it has a neighbors with label I gave to v one of his neighbor label. I iteratively continue the process until convergence (no more labels added).
an interesting outcome was the number of iteration as function of the p:

I check the zero idea on the new label I made and the results shown below:
when p_in = p_out = 0.5 the accuracy was 83%
when p_in = p_out = 0.1 the accuracy was 63%
when p_in = p_out = 0.001 the accuracy was 55%
