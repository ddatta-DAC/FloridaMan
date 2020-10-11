import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import Any
from queue import PriorityQueue
from collections import Counter
from sys import path

path.append('./..')


@dataclass(order=True)
class PrioritizedItem:
    value: int
    item: Any = field(compare=False)


# read corpus
def read_in_corpus(corpus_file=None):
    if corpus_file is None:
        corpus_file = './../data_preprocess/sentence_withIds.pkl'
    with open(corpus_file, 'rb') as fh:
        corpus = pickle.load(fh)
    all_sent = []

    for s in corpus:
        all_sent.extend(s)
    counter = Counter(all_sent)

    data = ((i, j) for i, j in zip(counter.keys(), counter.values()))
    corpus_df = pd.DataFrame(data, columns=['token', 'count'])
    corpus_df = corpus_df.sort_values(by=['count'], ascending=False)
    return corpus_df.reset_index(drop=True)


# Create huffman tree
class leaf_node:
    def __init__(self, token, freq):
        self.token = token
        self.path = ''
        self.is_leaf = True
        self.freq = freq


class node:
    def __init__(self, freq):
        self.path = ''
        self.is_leaf = False
        self.left = None
        self.right = None
        # frequency
        self.freq = freq


def create_huffman_tree(corpus_file=None):
    corpus_df = read_in_corpus(corpus_file)
    # Create a node for each token
    # Insert into the Queue
    Q = PriorityQueue()

    for i, row in corpus_df.iterrows():
        token = row['token']
        freq = row['count']
        obj = leaf_node(
            token,
            freq
        )

        PI_obj = PrioritizedItem(freq, obj)
        Q.put(PI_obj)

    root = None
    while Q.empty() is False:
        dq1 = Q.get()

        if Q.empty():
            root = dq1.item
            break

        dq2 = Q.get()
        f = dq1.item.freq + dq2.item.freq

        n = node(f)
        n.left = dq2.item
        n.right = dq1.item
        PI_obj = PrioritizedItem(f, n)
        Q.put(PI_obj)

    # Traverse the tree and assign paths
    start = ''

    def traverse(cur_node, _path):
        if cur_node is None: return
        cur_node.path = _path
        if cur_node.is_leaf: return
        traverse(cur_node.left, _path + '0')
        traverse(cur_node.right, _path + '1')
        return

    traverse(root, start)
    return root


def get_token2path_dict(tree):
    _dict = {}
    def traverse(cur_node):
        if cur_node is None:
            return _dict

        if cur_node.is_leaf:
            _dict[cur_node.token] = cur_node.path
            return _dict
        traverse(cur_node.left)
        traverse(cur_node.right)
        return _dict

    _dict = traverse(tree)
    return _dict


def get_huffman_tree(corpus_file=None):
    tree = create_huffman_tree()
    token2path_dict = get_token2path_dict(tree)
    return tree, token2path_dict
# tree = create_huffman_tree()
# token2path_dict = get_token2path_dict(tree)
