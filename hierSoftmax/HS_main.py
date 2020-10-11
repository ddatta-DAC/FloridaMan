import torch
from torch import nn
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from huffman_tree import get_huffman_tree
from torch import FloatTensor as FT
from torch import LongTensor as LT

tree, token2path_dict = get_huffman_tree()


def read_in_corpus(corpus_file=None):
    if corpus_file is None:
        corpus_file = './../data_preprocess/sentence_withIds.pkl'
    with open(corpus_file, 'rb') as fh:
        corpus = pickle.load(fh)
    return corpus


class word2vec_module(nn.Module):
    def __init__(self, emb_dim, token2path_dict):
        super(word2vec_module, self).__init__()
        self.vocab_size = len(token2path_dict)
        self.V = max([len(_) for _ in token2path_dict.values()])
        self.emb_layer = nn.Embedding(vocab_size + 1, emb_dim)
        nn.init.uniform_(emb_layer.weight, -1.0, 1.0)
        self.W_matrix = torch.nn.parameter.Parameter(
            FT(np.random.random([V, emb_dim]))
        )
        nn.init.uniform_(self.W_matrix, -1.0, 1.0)
        self.token2path_dict = token2path_dict
        return

    def get_paths(self, word_ids):
        res = []
        for s in word_ids:
            s = s.data.numpy().tolist()
            s = list(token2path_dict[s])
            z = list(map(lambda x: np.float(x), s))
            res.append(np.array(z))
        return res

    def calculate_P(self, paths):
        # replace 0 with -1
        for p in paths:
            p[p == 0] = -1

        # Pad with 0's
        z = [np.zeros(self.V - len(_)) for _ in paths]
        padded_paths = [np.concatenate([i, j]) for i, j in zip(paths, z)]
        padded_paths = FT(padded_paths)

        pp_mask = padded_paths.data.numpy().copy()
        pp_mask[pp_mask == -1] = 1
        pp_mask = FT(pp_mask)
        return padded_paths, pp_mask

    def forward(self, target, context):

        c_vec = self.emb_layer(context)
        paths = self.get_paths(target)

        # padded path has shape[batch, max_len_path]
        # pp_mask masks out the last part of the path (truncate)
        padded_paths, pp_mask = self.calculate_P(paths)

        x1 = torch.tensordot(c_vec, torch.transpose(self.W_matrix, 1, 0), 1)
        # +1 / -1 to tell which direction one is going in the tree
        nc = c_vec.shape[1]
        padded_paths = padded_paths.unsqueeze(1).repeat(1, nc, 1)
        pp_mask = pp_mask.unsqueeze(1).repeat(1, nc, 1)

        x2 = x1 * padded_paths
        # -----------
        # Take sigmoid
        # -----------
        x3 = torch.sigmoid(x2)

        # ------------
        # Take log
        # ------------
        x4 = torch.log(x3)

        # -----------
        # Mask invalid path segments
        # -----------
        x5 = x4 * FT(pp_mask)
        loss_val = torch.sum(x5, dim=-1, keepdims=False)
        loss_val = torch.sum(loss_val, dim=-1, keepdims=False)
        loss_val = -torch.mean(loss_val, dim=-1, keepdims=False)
        return loss_val


corpus = read_in_corpus()
context_size = 1
train_words = []
train_contexts = []
for sentence in corpus:
    words = []
    context = []
    len_s = len(sentence)
    for pos in range(context_size, len_s - context_size):
        words.append(sentence[pos])
        tmp = []
        for c in range(context_size):
            tmp.append(sentence[pos + (c + 1)])
            tmp.append(sentence[pos - (c + 1)])
        context.append(tmp)
    train_words.extend(words)
    train_contexts.extend(context)
train_contexts = np.array(train_contexts)
train_words = np.array(train_words)

# ================================================== #

w2v_object = word2vec_module(300, token2path_dict)

def train_model(
        w2v_object,
        words,
        contexts,
        LR=0.01,
        num_epochs=10,
        batch_size=512
):
    opt = torch.optim.Adam(list(w2v_object.parameters()), LR)
    opt.zero_grad()
    idx = np.arange(words.shape[0], dtype=int)

    bs = batch_size
    for epochs in tqdm(range(num_epochs)):
        np.random.shuffle(idx)
        num_batches = idx.shape[0] // batch_size + 1
        for b in range(num_batches):
            b_idx = idx[b * bs:(b + 1) * bs]
            b_idx = b_idx.astype(int)

            b_w = LT(words[b_idx])
            b_c = LT(contexts[b_idx.astype(int)])

            opt.zero_grad()
            # take each sentence
            _loss = w2v_object(b_w, b_c)
            _loss.backward()
            opt.step()
            if (b + 1) % 200 == 0:
                print('Batch {} Loss {:4f}'.format(b + 1, np.mean(_loss.data.numpy())))
    return w2v_object


w2v_object = train_model(
    w2v_object,
    train_words,
    train_contexts,
    0.01
)
