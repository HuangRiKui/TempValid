import pdb
import pickle
import json
import datetime
import logging
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from grapher import Grapher
from dataloader import load_data, Dataset
import rule_application as ra
from models import LCWT, LCFT, LTV, Noisy_OR, TempValid

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="ICEWS14", type=str)
parser.add_argument("--model", "-m", default="TempValid", type=str)  # LCWT: without time, LCFT: static time, LTV: learning time, Noisy_OR: Noisy_OR score
parser.add_argument("--lr", "-lr", default=1e-2, type=float)
parser.add_argument("--alpha", "-a", default=1.0, type=float)
parser.add_argument("--beta", "-b", default=0.1, type=float)
parser.add_argument("--rate", "-r", default=13, type=int)
parser.add_argument("--neg_num", "-n", default=100, type=int)
parser.add_argument("--max_epoch", "-e", default=3000, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--valid_epoch", "-v", default=20, type=int)
parser.add_argument("--seed", "-s", default=3407, type=int)
parser.add_argument("--cuda", action = "store_true")
parser.add_argument("--ta", action = "store_true") # time-aware negative sample generating
parser.add_argument("--save_name", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=[1,2,3], type=int, nargs="+")

# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logger(args):
    """Write logs to checkpoint and console"""

    set_name = '{0}'.format(args.model)
    if args.save_name!='':
        set_name = args.save_name + '_' + set_name

    log_dir = '../output/{0}/results/{1}/'.format(args.dataset, set_name)
    model_dir = '../output/{0}/results/{1}/models/'.format(args.dataset, set_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_file = log_dir + '{}.log'.format(set_name)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return model_dir

def conf_learning(args, rel, train_data, valid_data, conf_tensor=None):
    train_num = len(train_data)
    rule_dim = train_data[0].shape[1]
    train_dataset = Dataset(train_data, args, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_num = 0
    if valid_data!= None and len(valid_data) >0:
        valid_num = len(valid_data)
        valid_dataset = Dataset(valid_data, args, split='valid')
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        valid_loader = None
    logging.info('rel_id: {0}, rule_num:{1}, train_num:{2}, valid_num:{3}'.format(rel, rule_dim - 2, train_num, valid_num))

    if args.model == 'LCWT':
        model = LCWT(rule_dim, args)
    elif args.model == 'LCFT':
        model = LCFT(rule_dim, args)
    elif args.model == 'LTV':
        model = LTV(rule_dim, conf_tensor, args)
    elif args.model == 'Noisy_OR':
        model = Noisy_OR(rule_dim, args)
    else:
        model = TempValid(rule_dim, args)

    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    min_valid_loss = 1e6
    min_train_loss = 1e6
    max_mrr = 0

    train_model_dict = {}
    valid_model_dict = {}
    # logging.info('Starting training rel{0}'.format(rel))
    decay_count = 0
    break_count = 0
    lr = args.lr
    with tqdm(total=args.max_epoch) as _tqdm:
        for epoch in range(args.max_epoch):
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.max_epoch))
            model = model.train()
            epoch_loss = 0
            epoch_num = 0
            for i_batch, (rel_data, neg_masks) in enumerate(train_loader):
                if args.cuda:
                    rel_data = rel_data.cuda()
                    neg_masks = neg_masks.cuda()
                model.zero_grad()
                pos_scores, neg_scores = model(rel_data)

                pos_scores = torch.exp(pos_scores/args.alpha)
                neg_scores = torch.exp(neg_scores/args.alpha)
                neg_scores = torch.sum(neg_scores * neg_masks, dim=-1) / (torch.sum(neg_masks, dim=1) + 1e-9)
                neg_scores.clamp_(min=1)
                loss = - torch.sum(torch.log(pos_scores / neg_scores))

                # only pos
                # pos_loss = pos_scores
                # loss = -torch.sum(pos_loss)

                loss.backward()
                optimizer.step()

                for name, para in model.named_parameters():
                    if 'W' in name:
                        para.data.clamp_(0,1)
                    if 'beta' in name:
                        para.data.clamp_(min=0)
                    else:
                        para.data.clamp_(min=0)

                epoch_loss += loss.item()
                epoch_num += rel_data.shape[0]

            epoch_loss /= epoch_num

            if epoch_loss < min_train_loss:
                min_train_loss = epoch_loss
                train_model_dict = model.state_dict().copy()
                decay_count = 0
            else:
                decay_count += 1

            if decay_count > 25:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                    lr = param_group['lr']
                decay_count = 0


            if (epoch+1)%args.valid_epoch ==0:
                # logging.info('Epoch {0}, train loss = {1}'.format(epoch+1 ,epoch_loss))
                if valid_loader != None:
                    ranks, valid_loss = evaluate(args, valid_loader, model)
                    valid_mrr = torch.mean(1/ranks).item()
                    if max_mrr < valid_mrr:
                        break_count = 0
                        max_mrr = valid_mrr
                        valid_model_dict = model.state_dict().copy()
                    else:
                        break_count +=1

            _tqdm.update(1)
            _tqdm.set_postfix(train_loss='{:.3e}'.format(epoch_loss), valid_mrr='{:.5f}'.format(max_mrr),
                              break_count = str(break_count), lr='{:.3e}'.format(lr))
            if break_count>20:
                print("early_stopping!")
                break
    logging.info('train_loss: {0}, valid_mrr:{1} \n'.format(round(epoch_loss,4), round(max_mrr,5)))

    if valid_model_dict == {}:
        model_dict = train_model_dict
    else:
        model_dict = valid_model_dict

    return model_dict

def evaluate(args, test_loader, model, mode='mrr'):
    model = model.eval()
    valid_loss = 0
    epoch_num = 0
    rank_list = []
    for i_batch, (rel_data, neg_masks) in enumerate(test_loader):
        if args.cuda:
            rel_data = rel_data.cuda()
            neg_masks = neg_masks.cuda()
        pos_scores, neg_scores = model(rel_data)
        pos_scores = pos_scores.reshape(neg_scores.shape[0],-1)

        neg_scores = neg_scores * neg_masks
        # put pos behind neg to avoid unreasonable rank under same score
        scores = torch.cat([neg_scores, pos_scores], dim=1)
        sort_scores = torch.argsort(scores, dim=1, descending=True)
        ranks = (sort_scores == sort_scores.shape[1] - 1).nonzero()[:, 1] + 1

        rank_list.append(ranks)

    if epoch_num>0:
        valid_loss /= epoch_num

    if mode=='mrr':
        rank_list = torch.cat(rank_list, dim=0)

    return rank_list, valid_loss

def trainer():
    for rel in rels:
        train_data = load_data(args, rel, split='train')
        valid_data = load_data(args, rel, split='valid')
        # train_rule
        if train_data != None and len(train_data) > 0:
            train_num = 2 ** args.rate
            train_data = train_data[:train_num]

            if rel in conf_tensor_dict:
                conf_tensor = conf_tensor_dict[rel]
            else:
                conf_tensor = torch.zeros(train_data.shape[1] - 1)
            rel_model_dict = conf_learning(args, rel, train_data, valid_data, conf_tensor)
            model_path = model_dir + 'rel_{}.pth'.format(rel)
            # model_dict[rel] = rel_model_dict
            torch.save(rel_model_dict, model_path)

def evaluation():
    eval_array = np.zeros(4)
    cover_count = 0
    all_count = 0
    for rel in rels:
        test_data = load_data(args, rel, split='test')
        test_data_num = quadruple_data.test_idx[quadruple_data.test_idx[:, 1] == rel].shape[0]
        all_count += test_data_num
        model_path = model_dir + 'rel_{}.pth'.format(rel)
        if test_data != None and len(test_data) > 0 and os.path.isfile(model_path):
            test_num = len(test_data)
            cover_count += test_num
            test_dataset = Dataset(test_data, args, split='test')
            test_loader = DataLoader(test_dataset, batch_size=32)
            rule_dim = test_data[0].shape[1]

            if args.model == 'LCWT':
                model = LCWT(rule_dim, args)
            elif args.model == 'LCFT':
                model = LCFT(rule_dim, args)
            elif args.model == 'LTV':
                model = LTV(rule_dim, conf_tensor, args)
            elif args.model == 'Noisy_OR':
                model = Noisy_OR(rule_dim, args)
            else:
                model = TempValid(rule_dim, args)

            rel_model_dict = torch.load(model_path)
            model.load_state_dict(rel_model_dict)

            if args.cuda:
                model = model.cuda()

            ranks, valid_loss = evaluate(args, test_loader, model, mode='mrr')
            mrr = torch.sum(1.0 / ranks).item()
            h1 = torch.sum(ranks <= 1).item()
            h3 = torch.sum(ranks <= 3).item()
            h10 = torch.sum(ranks <= 10).item()
            eval_array += np.array([mrr, h1, h3, h10])
            logging.info('rel_id: {}, MRR: {:.4f}, H@1: {:.4f}, H@3: {:.4f}, H@10: {:.4f}, test_num: {}'.format(
                rel, mrr / test_num, h1 / test_num, h3 / test_num, h10 / test_num, test_num))
        else:
            logging.info('rel_id: {}, MRR: {:.4f}, H@1: {:.4f}, H@3: {:.4f}, H@10: {:.4f}, test_num: {}'.format(
                rel, test_data_num, 0, 0, 0, 0))
        logging.info('Accum: MRR: {:.4f}, H@1: {:.4f}, H@3: {:.4f}, H@10: {:.4f}, count:{}/{}'.format(eval_array[0] / all_count, eval_array[1] / all_count, eval_array[2] / all_count, eval_array[3] / all_count, all_count, all_test_num))
        logging.info('-------------------------------------------------------------------------------------')
        logging.info('                                                                                     ')

    return eval_array


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = set_logger(args)
    same_seeds(args.seed)
    logging.info(args)
    dataset_dir = "../data/" + args.dataset + "/"
    quadruple_data = Grapher(dataset_dir)
    rels = list(quadruple_data.id2relation.keys())
    rels.sort()
    args.ent_num = len(quadruple_data.entity2id)
    all_test_num = len(quadruple_data.test_idx)
    rules_dict = json.load(open("../output/"+args.dataset+"/rules_dict.json"))
    rules_dict = {int(k): v for k, v in rules_dict.items()}
    rule_lengths = [args.rule_lengths] if (type(args.rule_lengths) == int) else args.rule_lengths
    rules_dict = ra.filter_rules(rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths)

    # record confidence learned by TLogic for LTV
    conf_tensor_dict = {}
    for rel_id in rules_dict.keys():
        rule_num = len(rules_dict[rel_id])
        conf_tensor = torch.zeros(rule_num)
        for rule_id, rule in enumerate(rules_dict[rel_id]):
            conf_tensor[rule_id] = rule['conf']
        conf_tensor_dict[rel_id] = conf_tensor

    trainer()
    eval_array = evaluation()

    logging.info('MRR:  {:.5f}'.format(eval_array[0]/all_test_num))
    logging.info('H@1:  {:.5f}'.format(eval_array[1]/all_test_num))
    logging.info('H@3:  {:.5f}'.format(eval_array[2]/all_test_num))
    logging.info('H@10: {:.5f}'.format(eval_array[3]/all_test_num))
