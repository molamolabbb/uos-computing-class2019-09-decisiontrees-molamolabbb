#!/usr/bin/env python

from math import log2

def is_tree(thing):
    if isinstance(thing, list) and (len(thing)==3) and isinstance(thing[0],tuple):
        return True
    else : False

def classify(tree, data):
    if is_tree(tree) is True:
        if data[tree[0][0]]>tree[0][1] : 
            return classify(tree[2],data)
        else : 
            return classify(tree[1],data)
    else : 
        return tree

def tree_accuracy(tree,x,y):
    correct = 0
    for x_i,y_i in zip(x,y) : 
        if classify(tree,x_i) == y_i :
            correct += 1
    return float(correct)/len(y)

# --- entropy ---

def entropy(class_probabilities):
    for p in class_probabilities:
        if p == 0:
            class_probabilities.remove(0)
    return -sum([class_probabilities[j]*log2(class_probabilities[j]) for j in range(len(class_probabilities))])

def class_probabilities(labels):    
    pro = [0 for i in range(max(labels))]
    for label in labels:
        pro[label-1] +=1
    tot = len(labels)
    return [p/tot for p in pro]

def data_entropy(labeled_data):
    labels = [y for x,y in labeled_data]
    print (labels)
    return entropy(class_probabilities(labels))

def partition_entropy(subsets):
    tot = sum(len(subset) for subset in subsets)
    tot_ent = 0
    for subset in subsets:
        q = float(len(subset))/float(tot)
        tot_ent += q*data_entropy(subset)
    return tot_ent



