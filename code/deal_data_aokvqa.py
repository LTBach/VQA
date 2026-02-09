from config import cfg
import os.path as osp
import pickle
import json
import pdb
import re
from utils import dele_a, transfer, hand_remove, deal_fact
from collections import defaultdict
import tqdm
# import Levenshtein
# import wordninja
from data import fvqa, preprocess
import random
import numpy as np
from collections import defaultdict


class Runner:
    def __init__(self, args):
        self.args = args
        self.data_path = osp.join(args.data_root, "aokvqa")

        self.exp_path = osp.join(self.data_path, "exp_data")
        self.common_path = osp.join(self.data_path, "common_data")

        self.raw_path = osp.join(self.exp_path, "raw")

        self.train_path = osp.join(self.raw_path, "aokvqa_v1p0_train.json")
        self.val_path = osp.join(self.raw_path, "aokvqa_v1p0_val.json")
        self.test_path = osp.join(self.raw_path, "aokvqa_v1p0_test.json")

        self.vocab_ans_500_path = osp.join(self.common_path, "answer.vocab.aokvqa.500.json")

        self.dump_train_path = osp.join(self.exp_path, "train_data")
        self.dump_val_path = osp.join(self.exp_path, "val_data")
        # self.dump_test_path = osp.join(self.exp_path, "test")

        self.dump_train_500_path = osp.join(self.dump_train_path, "all_qs_dict_release_train_500.json")
        self.dump_val_500_path = osp.join(self.dump_val_path, "all_qs_dict_release_val_500.json")

    # def statistics_of_ans_and_entity(self, path=None):
    #     with open(path, 'r') as fp:
    #         dic = json.load(fp)
    #         ans_set = set()
    #         entity_set = set()
    #         relation_set = set()
    #         dic_len = 0
    #         for key in dic.keys():
    #             dic_len += 1
    #             e1 = dic[key]["fact"][0]
    #             r = dic[key]["fact"][1]
    #             e2 = dic[key]["fact"][2]
    #             ans = dic[key]["answer"]
    #             ans_set.add(ans)
    #             entity_set.add(e1)
    #             entity_set.add(e2)
    #             relation_set.add(r)

    #         ans_or_entity = ans_set | entity_set
    #         ans_and_entity = ans_set & entity_set
    #         print("ans_set len:", len(ans_set))
    #         print("entity_set len:", len(entity_set))
    #         print("ans_or_entity len:", len(ans_or_entity))
    #         print("ans_and_entity len:", len(ans_and_entity))
    #         print("relation len:", len(relation_set))
    #         print("dic len:", dic_len)

    def create_filtered_data(self):
        self.filter_data(self.train_path, self.dump_train_500_path)
        self.filter_data(self.val_path, self.dump_val_500_path)


    def filter_data(self, path=None, dump_path=None):
        with open(path, 'r') as fp:
            data = json.load(fp)

        with open(self.vocab_ans_500_path, 'r') as fp:
            dic = json.load(fp)

        ans_500_list = dic['answer'].keys()
        new_data = []

        for idx in range(len(data)):
            for ans in data[idx]['choices']:
                if ans in ans_500_list:
                    new_data.append(data[idx])

        with open(dump_path, 'w') as fp:
            json.dump(new_data, fp)




if __name__ == '__main__':
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    runner = Runner(cfg)

    runner.create_filtered_data()
