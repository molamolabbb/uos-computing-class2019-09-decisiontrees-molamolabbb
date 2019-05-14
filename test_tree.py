from tree import (is_tree, classify, tree_accuracy,
                  class_probabilities, data_entropy, entropy,
                  partition_entropy)
from pytest import approx
from math import log2

def test_classify():
    assert not is_tree([1, 2, 3, 4])
    assert not is_tree([1, 2, 3])
    assert not is_tree([1, 2])
    assert not is_tree([1])
    assert not is_tree([])
    assert not is_tree(0.5)
    assert not is_tree(0)
    assert is_tree([(0,0), [], []])
    assert is_tree([(1,0.5), [(1,1), 1, 2], [(1,1), 1, 2]])
    assert is_tree([(1,0.5), 1, 2])
    assert is_tree([(1,0.5), 1, [(1,1), 1, 2]])

def test_classify():
    assert classify(1, [1]) == 1
    assert classify(2, [1]) == 2
    assert classify([(0, 0.5), 1, 2], [1]) == 2
    assert classify([(0, 0.5),
                      1,
                      [(0, 1.5), 2, 3]],
                    [1]) == 2
    assert classify([(0, 0.5),
                      1,
                      [(1, 1.5), 2, 3]],
                    [1, 2, 3]) == 3
    assert classify([(0, 0.5),
                      1,
                      [(1, 1.5),
                       2,
                       [(2, 4.5), 3, 4]]],
                    [1, 2, 3]) == 3
    atree = [(0, 0.5),
             1,
             [(1, 1.5),
              [(2, 2.1), 5, 6],
              [(2, 4.5), 3, 4]]]
    assert classify(atree, [1, 2, 3]) == 3
    assert classify(atree, [1, 1.1, 3]) == 6
    assert classify(atree, [1, 1.1, 1.1]) == 5
    assert classify(atree, [0.2, 1.1, 1.1]) == 1
    assert classify(atree, [2.2, 1.7, 5.1]) == 4

def test_tree_accuracy():
    tree = [(0, 0.5), 1, 2]
    assert tree_accuracy(tree, [[0], [1]], [1, 2]) == approx(1.0)
    assert tree_accuracy(tree, [[0], [1]], [2, 1]) == approx(0.0)
    atree = [(0, 0.5),
             1,
             [(1, 1.5),
              [(2, 2.1), 5, 6],
              [(2, 4.5), 3, 4]]]
    x = [[1, 2, 3],
         [1, 1.1, 3],
         [1, 1.1, 1.1],
         [0.2, 1.1, 1.1],
         [2.2, 1.7, 5.1]]
    y = [3, 6, 5, 1, 4]
    assert tree_accuracy(atree, x, y) == approx(1.)
    y = [3, 6, 5, 1, 3]
    assert tree_accuracy(atree, x, y) == approx(4./5.)

def test_entropy():
    assert entropy([1]) == -1.0 * log2(1.0)
    assert entropy([1]) == -1.0 * log2(1.0)
    assert entropy([0.5, 0.5]) == -0.5 * log2(0.5) -0.5 * log2(0.5)
    assert entropy([0.25, 0.25, 0.5]) == - 0.25 * log2(0.25) - 0.25 * log2(0.25) - 0.5 * log2(0.5)

def test_class_probabilities():
    assert class_probabilities([1]) == [1.0]
    assert class_probabilities([1, 1, 1]) == [1.0]
    assert class_probabilities([1, 2]) == [0.5, 0.5]
    assert class_probabilities([1, 2, 2, 1]) == [0.5, 0.5]
    assert class_probabilities([1, 2, 1, 2]) == [0.5, 0.5]
    assert class_probabilities([1, 3, 2, 1]) in [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]

def test_data_entropy():
    assert data_entropy([ [[], 1] ]) == -1.0 * log2(1.0)
    assert data_entropy([ [[], 1], [[], 1] ]) == -1.0 * log2(1.0)
    assert data_entropy([ [[], 1], [[], 2] ]) == -0.5 * log2(0.5) -0.5 * log2(0.5)

def test_partition_entropy():
    assert partition_entropy([ [ [[], 1] ] ]) == -1.0 * log2(1.0)
    assert partition_entropy([ [ [[], 1] ], [ [[], 2] ] ]) == 0
    assert partition_entropy([ [ [[], 1], [[], 2] ] ]) == -0.5 * log2(0.5) + -0.5 * log2(0.5)
    assert partition_entropy([ [ [[], 1], [[], 2] ],
                               [ [[], 1], [[], 1] ] ]) == 0.5* (-0.5 * log2(0.5) + -0.5 * log2(0.5)) + 0.5*0
