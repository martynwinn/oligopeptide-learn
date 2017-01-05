# data are 7 aa long sequences
# aim to classify as alpha or beta


# Standard scientific Python imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, tree, metrics
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

verbose = False
classifier_type="DecisionTree"
#classifier_type="RandomForest"
#classifier_type="NearestNeighbour"
#classifier_type="SVM"
#classifier_type="NeuralNet"
allowed_residues = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

def ShowDecisionTree(estimator, X_test):

    ''' Taken from 
    http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    '''

    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("\nThe binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    if verbose:
        for i in range(n_nodes):
            if is_leaves[i]:
                print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            else:
                print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
                      "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
        for f in feature:
            if f > -1:
                resno = f / 20 + 1
                print "feature "+str(f)+": is residue "+str(resno)+" a "+allowed_residues[f % 20]

    print "Maximum depth %d ",np.amax(node_depth)
    print "Number of leaves %d ",np.count_nonzero(is_leaves)
    print "Number of decision nodes %d ",n_nodes - np.count_nonzero(is_leaves)

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

    print "\nNode path for sample "+str(sample_id)+ " is "+ str(node_index)

    # For a group of samples, we have the following common node.
    sample_ids = [0, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
      % (sample_ids, common_node_id))
    print("It is %s %% of all nodes.\n" % (100 * len(common_node_id) / n_nodes,))

### start of main

infile = "ss_list_A_top100k.txt"
with open(infile) as f:
    data = []
    target = []
    for line in f:
        words = line.split()
        data.append(words[0])
        target.append(words[1])

nitems = len(data)
print "Total number of sequences = "+str(nitems)
print data[nitems/2], target[nitems/2]

# The following converts a 7-residue oligopeptide string into a 140 binary feature vector
print "binarizing"

binary_data = []
for oligopeptide in data:
    #if oligopeptide[0:2] == 'AA':
    #    print oligopeptide
    peptides = list(oligopeptide)
    binary_data.append(label_binarize(peptides, classes=allowed_residues).reshape(-1))
print binary_data[nitems/2], target[nitems/2]

print "Training"

# Create a classifier: a support vector classifier
if classifier_type == "DecisionTree":
    classifier = tree.DecisionTreeClassifier(max_depth=50,min_samples_leaf=1)
elif classifier_type == "RandomForest":
    classifier = RandomForestClassifier(n_estimators=10)
elif classifier_type == "NearestNeighbour":
    classifier = NearestNeighbors()
elif classifier_type == "SVM":
    classifier = svm.SVC()
elif classifier_type == "NeuralNet":
    classifier = MLPClassifier()

# We learn the digits on the first half of the digits
classifier.fit(binary_data[:nitems / 2], target[:nitems / 2])

if classifier_type == "DecisionTree":
    print "Show Decision Tree"
    ShowDecisionTree(classifier, binary_data[nitems / 2: nitems])

print "Predicting"

# Now predict the value of the digit on the second half:
expected = target[nitems / 2: nitems]
predicted = classifier.predict(binary_data[nitems / 2: nitems])
score = classifier.score(binary_data[nitems / 2: nitems],  target[nitems / 2: nitems])

print "Predicting "+str(len(expected))+" items with score " + str(score)

print 'Output statistics'

outfile = open("out.txt",'w')
outfile.write("Sequence     Expected     Predicted\n")

exp = [0,0]
pred_true = [0,0]
pred_false = [0,0]
for i in range(len(expected)):
    outfile.write(data[i+nitems/2]+' '+str(expected[i])+' '+str(predicted[i])+'\n')
    if expected[i] == 'H':
        exp[0] += 1
        if predicted[i] == 'H':
            pred_true[0] += 1
        elif predicted[i] == 'E':
            pred_false[0] += 1
    elif expected[i] == 'E':
        exp[1] += 1
        if predicted[i] == 'E':
            pred_true[1] += 1
        elif predicted[i] == 'H':
            pred_false[1] += 1

print pred_true, pred_false

print "True helix rate",pred_true[0]/float(pred_true[0]+pred_false[0])
print "False strand rate",pred_false[0]/float(pred_false[0]+pred_true[0])
print "True strand rate",pred_true[1]/float(pred_true[1]+pred_false[1])
print "False helix rate",pred_false[1]/float(pred_false[1]+pred_true[1])

#Plot results

index = np.arange(2)
bar_width = 0.25
rects1 = plt.bar(index, exp, bar_width,
                 alpha=0.4,
                 color='b',
                 label='Expected')
rects2 = plt.bar(index+bar_width, pred_true, bar_width,
                 alpha=0.4,
                 color='g',
                 label='Correctly predicted')
rects3 = plt.bar(index+2*bar_width, pred_false, bar_width,
                 alpha=0.4,
                 color='r',
                 label='Incorrectly predicted')

plt.legend()
#plt.show()
