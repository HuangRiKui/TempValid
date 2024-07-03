import json
import pdb
import random
import time
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import pickle
import os

import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
from rule_learning import rules_statistics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--split", "-s", default="test", type=str)
parser.add_argument("--rules", "-r", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=1, type=int, nargs="+")
parser.add_argument("--window", "-w", default=-1, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--neg_num", "-n", default=100, type=int)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--seed", default=0, type=int)

parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
neg_num = parsed["neg_num"]
num_processes = parsed["num_processes"]
rule_lengths = parsed["rule_lengths"]
split = parsed['split']
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
seed = parsed['seed']
np.random.seed(seed)
random.seed(seed)

dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
output_dir = dir_path + split + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
data = Grapher(dataset_dir)
learn_edges = store_edges(data.train_idx)

rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
rules_statistics(rules_dict)
print(' ')
rules_dict = ra.filter_rules(rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths)
print("Rules statistics after pruning:")
rules_statistics(rules_dict)

def create_data(i, num_queries, neg_num=30, split='test'):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    output_array_dict = {}

    num_rest_queries = len(shuffle_list) -  num_processes * num_queries


    if num_rest_queries > i:
        process_queries_idx = shuffle_list[i * (num_queries+1): (i + 1) * (num_queries+1)]
    else:
        process_queries_idx = shuffle_list[num_rest_queries * (num_queries+1) + (i-num_rest_queries) * num_queries: num_rest_queries * (num_queries+1) + (i-num_rest_queries+1) * num_queries]

    it_start = time.time()
    for j_id, j in enumerate(process_queries_idx):
        quadruple = process_data[j]
        sub = quadruple[0]
        rel = quadruple[1]
        obj = quadruple[2]
        cur_ts = quadruple[3]
        if rel in rules_dict:
            rule_num = len(rules_dict[rel])
        else:
            continue
        query_data = process_data[process_data[:,1]==rel]
        pos_ent_id = (query_data[:,0]==sub)&(query_data[:,1]==rel)&(query_data[:,3]==cur_ts)
        pos_cands = set(query_data[pos_ent_id][:,2])
        edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)
        cands_dict = {}
        pos_array = np.zeros(rule_num+1, dtype=np.int16)
        # t1 = time.time()
        for rule_id, rule in enumerate(rules_dict[rel]):
            walk_edges = ra.match_body_relations(rule, edges, quadruple[0])
            if 0 not in [len(x) for x in walk_edges]:
                rule_walks = ra.get_walks(rule, walk_edges)
                if rule["var_constraints"]:
                    rule_walks = ra.check_var_constraints(
                        rule["var_constraints"], rule_walks
                    )

                if not rule_walks.empty:
                    # get cands
                    cands_dict = ra.get_grounding_candidates(
                        rule,
                        rule_walks,
                        cur_ts,
                        cands_dict,
                        rule_id,
                        rule_num
                    )
                rule_walks={}
        # print(time.time()-t1)

        # pos_array
        if obj in cands_dict:
            pos_array = cands_dict[obj]
            pos_array[-1] = 1

        # filter other positive cand
        cands_dict = {cand: cand_array for cand, cand_array in cands_dict.items() if cand not in pos_cands}
        # neg sampling by tlogic neg cand

        # normal sample
        if split != 'test':
            cands_score = {}
            # rule-adversarial
            for cand, cand_array in cands_dict.items():
                # Score by TLogic
                array_mask = cand_array[:-1] == 0
                time_score = 0.5 * np.exp(-0.1 * cand_array[:-1])
                conf_score = 0.5 * conf_array_dict[rel][:-1]
                score = time_score + conf_score
                score[array_mask] = 0
                score = score[score != 0]

                top_20_idx = score.argsort()[::-1][0:20]
                score = score[top_20_idx]
                score = 1 - np.product(1 - score)
                cands_score[cand] = score
            sorted_dict = sorted(cands_score.items(), key=lambda x: x[1], reverse=True)
            neg_cand_list = [cand_tuple[0] for cand_tuple in sorted_dict[:neg_num]]
            sorted_dict = {}

            neg_cand_arr = np.array([cands_dict[neg_cand] for neg_cand in neg_cand_list], dtype=np.int16)
            actual_cand_num = len(neg_cand_list)
        else:
            neg_cand_arr = np.array([cands_dict[neg_cand] for neg_cand in cands_dict], dtype=np.int16)
            actual_cand_num = neg_cand_arr.shape[0]
        if actual_cand_num < neg_num:
            pad_arr = np.zeros((neg_num - actual_cand_num, rule_num + 1), dtype=np.int16)
            neg_cand_arr = neg_cand_arr.reshape(-1, rule_num + 1)
            neg_cand_arr = np.concatenate((neg_cand_arr, pad_arr), axis=0)

        pos_array = pos_array.reshape(1, -1)
        output_array = np.concatenate((pos_array, neg_cand_arr), axis=0)
        # save memory by sparse matrix
        row, col = np.nonzero(output_array)
        values = output_array[row, col]
        query_csr = csr_matrix((values, (row, col)), shape=output_array.shape)
        output_array_dict[j] = query_csr
        del query_csr
        del output_array

        if not (j_id + 1) % 20:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: {1} samples of rel {2} finished: {3}/{4}, {5} sec".format(
                    i, split, rel, j_id + 1, len(process_queries_idx), round(it_time,2)
                )
            )
            it_start = time.time()

    return output_array_dict

conf_array_dict = {}
for rel_id in rules_dict.keys():
    rule_num = len(rules_dict[rel_id])
    conf_array = np.zeros(rule_num+1)
    for rule_id, rule in enumerate(rules_dict[rel_id]):
        conf_array[rule_id] = rule['conf']
    conf_array_dict[rel_id] = conf_array

all_data = data.all_idx

if split == 'train':
    query_index_list = []
    train_data = data.train_idx
    process_data = data.train_idx
    for rel in rules_dict.keys():
        # limit size of training data
        rule_num = len(rules_dict[rel])
        multi_data = 30
        data_num = rule_num*multi_data
        query_index_list.extend(np.where(train_data[:, 1] == rel)[0][-data_num:].tolist())
    process_num = len(query_index_list)

elif split == 'valid':
    process_data = data.valid_idx
    process_num = len(process_data)
    query_index_list = list(range(process_num))
else:
    process_data = data.test_idx
    process_num = len(process_data)
    query_index_list = list(range(process_num))

rel_query_dict = {}
for query_index in query_index_list:
    query_rel = process_data[query_index][1]
    if query_rel not in rel_query_dict:
        rel_query_dict[query_rel] = []
    rel_query_dict[query_rel].append(query_index)

if split=='test':
    neg_num = len(data.id2entity)-1

count = 0
sort_keys = sorted(list(rel_query_dict.keys()))
max_rel = max(sort_keys)


pre_count = 0
start_rel = 0
end_rel = 1000
for rel in sort_keys:
    if rel >= start_rel and rel < end_rel:
        pre_count += len(rel_query_dict[rel])

for rel in sort_keys:
    filename = "{0}_{1}.pkl".format(split, rel)
    # if not os.path.isfile(output_dir + filename) and rel >=start_rel and rel <end_rel:
    if rel >= start_rel and rel < end_rel:
        # shuffle to balance cpu
        shuffle_list = rel_query_dict[rel]
        random.shuffle(shuffle_list)
        output_list = []
        output_wb = []

        print('Create {0} data of rel {1}, query num: {2}'.format(split, rel, len(shuffle_list)))
        rel_start_time = time.time()
        num_queries = len(shuffle_list) // num_processes
        output = Parallel(n_jobs=num_processes)(
            delayed(create_data)(i, num_queries, neg_num=neg_num, split=split) for i in range(num_processes)
        )

        if rel in rules_dict:
            for i in range(num_processes):
                for query_id in output[i].keys():
                    query_csr = output[i][query_id]
                    output_list.append(query_csr)

            count += len(output_list)

            pickle.dump(output_list, open(output_dir + filename, "wb"))
            rel_end_time = time.time()
            rel_use_time = (rel_end_time-rel_start_time)/60.0
            print('rel: {0}/{1} of {2} {3} data done! time use:{4} min'.format(rel, max_rel, dataset, split, round(rel_use_time,2)))
            print('schedule: {0} / {1}!'.format(count, pre_count))
            print('   ')
