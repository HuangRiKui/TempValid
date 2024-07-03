import os
import pdb
import pickle
import json
import numpy as np
import torch
from scipy.sparse import csr_matrix

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, rel_data, args, split):
        self.split = split
        self.args = args
        self.neg_num = self.args.neg_num
        self.ent_num = self.args.ent_num
        self.neg_num_list = []
        self.data, self.neg_mask = self.pre_process(rel_data)

    def __getitem__(self, index):
        if self.split == 'test':
            data = torch.from_numpy(self.data[index].toarray()).float()
            neg_mask = torch.from_numpy(self.neg_mask[index]).float()
        else:
            data = self.data[index]
            neg_mask = self.neg_mask[index]

        return data, neg_mask

    def __len__(self):
        return len(self.data)

    def pre_process(self, data):
        if self.split !='test':
            data = [data_item.toarray()[:self.neg_num+1] for data_item in data]
            if self.split == 'train':
                data = self.pad_train_negs(data)
            data = np.stack(data, axis=0)
            data = torch.from_numpy(data).contiguous().float()
            neg_mask = data[:,1:,-1]
            self.neg_num_list += neg_mask.sum(1).tolist()
            data = data[:, :, :-1]
        else:
            data, neg_mask = self.pad_test_negs(data)
        return data, neg_mask

    def pad_train_negs(self, data_list):
        output_list = []
        for data in data_list:
            actual_neg = sum(data[1:, -1])
            pad_num = self.neg_num - actual_neg
            if pad_num > 0:
                data = data[:actual_neg+1]
                if self.args.ta:
                    pad_array = np.repeat(data[0, :].reshape(1,-1), pad_num, axis=0)
                    neg_time = np.arange(1, pad_num+1).reshape(-1, 1)
                    pad_array = pad_array - neg_time
                    pad_array[pad_array<0] = 0
                    pad_label = np.sum(pad_array, axis=1)
                    pad_label[pad_label>0] = 1
                    pad_array[:,-1] = pad_label
                    data = np.concatenate((data, pad_array), axis=0)
                else:
                    pad_array = np.zeros((pad_num, data.shape[1]))
                    data = np.concatenate((data, pad_array), axis=0)

            output_list.append(data)
        return output_list

    def pad_test_negs(self, items):
        output_list = []
        neg_masks = []
        rule_dim = items[0].shape[1]
        neg_nums = [item.shape[0] for item in items]
        max_neg = self.args.ent_num
        for i, item in enumerate(items):
            neg_mask = np.ones(max_neg-1)
            item = item.toarray()
            neg_item = item[1:]
            neg_mask[:neg_item.shape[0]] = neg_item[:,-1]

            pad_num = max_neg - neg_nums[i]
            if pad_num > 0:
                pad_array = np.zeros((pad_num, rule_dim))
                pad_array = np.concatenate((item, pad_array), axis =0)

                neg_mask[-pad_num:] = 0
            else:
                pad_array = item
            pad_array = pad_array[:, :-1]
            row, col = np.nonzero(pad_array)
            values = pad_array[row, col]
            pad_csr = csr_matrix((values, (row, col)), shape=pad_array.shape)
            output_list.append(pad_csr)
            neg_masks.append(neg_mask)

        return output_list, neg_masks


def load_data(args, rel, split='train'):
    data_path = '../output/{0}/{1}/{1}_{2}.pkl'.format(args.dataset, split, rel)

    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None
