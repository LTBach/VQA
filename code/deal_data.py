from config import cfg
import os.path as osp
import pickle
import json
import pdb
import re
import nltk

from collections import Counter
from data.preprocess import process_punctuation

class Runner(object):
    def __init__(self, args):
        self.args = args
        self.data_path = osp.join(self.args.data_root, "aokvqa")
        self.exp_path = osp.join(self.data_path, "exp_data")
        self.common_path = osp.join(self.data_path, "common_data")
        self.raw_path = osp.join(self.exp_path, "raw")
        self.ans_list = []
        self.question_list = []
        self.relation_list = []
        self.crt_ans_list = []

        self.question_ans_dict = {}
        self.ans_question_dict = {}

        self.ans_relation_dict = {}
        self.relation_ans_dict = {}

        self.dump_ans_path = osp.join(self.common_path, "all_ans.txt")
        self.dump_question_path = osp.join(self.common_path, "all_question.txt")
        self.dump_relation_path = osp.join(self.common_path, "all_relation.txt")
        self.dump_crt_ans_path = osp.join(self.common_path, "all_crt_ans.txt")

        self.dump_vocab_ans_path = osp.join(self.common_path, "answer.vocab.aokvqa.json")
        self.dump_vocab_question_path = osp.join(self.common_path, "question.vocab.aokvqa.json")

        self.dump_vocab_ans_500_path = osp.join(self.common_path, "answer.vocab.aokvqa.500.json")
        self.dump_vocab_question_500_path = osp.join(self.common_path, "question.vocab.aokvqa.500.json")

        self.dump_vocab_ans_relation_500_path = osp.join(self.common_path, "question.vocab.aokvqa.relation.500.json")

    def get_list(self, path=None):
        path = osp.join(self.raw_path, path)
        with open(path, 'r') as fp:
            data = json.load(fp)
            for i in range(len(data)):
                self.ans_list.extend(data[i]['choices'])
                self.question_list.append(data[i]['question'])
                self.relation_list.extend(data[i]['rationales'])
                crt_idx = int(data[i]['correct_choice_idx'])
                self.crt_ans_list.append(data[i]['choices'][crt_idx])
                
                self.question_ans_dict[data[i]['question']] = data[i]['choices']

                for ans in data[i]['choices']:
                    for relation in data[i]['rationales']:
                        if self.ans_relation_dict.get(ans) is None:
                            self.ans_relation_dict[ans] = [relation]
                        else:
                            self.ans_relation_dict[ans].extend([relation])


        print('Number of:')
        print('   ans:', len(self.ans_list))
        print('   question:', len(self.question_list))
        print('   relation:', len(self.relation_list))

        for ans in set(self.ans_list):
            questions = []
            for question in set(self.question_list):
                if ans in self.question_ans_dict[question]:
                    questions.append(question)
            self.ans_question_dict[ans] = questions

        for relation in set(self.relation_list):
            anss = []
            for ans in set(self.ans_list):
                if ans in self.ans_relation_dict[ans]:
                    anss.append(ans)
            self.relation_ans_dict[relation] = anss


    def dump_all(self, path=None):

        self._dump_ans(self.dump_ans_path)
        self._dump_question(self.dump_question_path)
        self._dump_relation(self.dump_relation_path)
        self._dump_crt_ans(self.dump_crt_ans_path)

    def _dump_ans(self, path=None):
        with open(path, 'w') as f:
            for ans in self.ans_list:
                f.write(ans)
                f.write('\n')

    def _dump_question(self, path=None):
        with open(path, 'w') as f:
            for question in self.question_list:
                f.write(question)
                f.write('\n')

    def _dump_relation(self, path=None):
        with open(path, 'w') as f:
            for relation in self.relation_list:
                f.write(relation)
                f.write('\n')

    def _dump_crt_ans(self, path=None):
        with open(path, 'w') as f:
            for crt_ans in self.crt_ans_list:
                f.write(crt_ans)
                f.write('\n')

    def dump_vocab(self, path=None):
        self.dump_vocab_ans(self.dump_vocab_ans_path)
        self.dump_vocab_question(self.dump_vocab_question_path)

        self.dump_vocab_ans_500(self.dump_vocab_ans_500_path)
        self.dump_vocab_question_500(self.dump_vocab_question_500_path)
        self.dump_vocab_ans_vocab_relation_500(self.dump_vocab_ans_relation_500_path)

    def dump_vocab_ans(self, path=None):
        ans_counter = Counter(self.ans_list).most_common()
        anss = [ans for ans, count in ans_counter]

        stoi = {ans: i for i, ans in enumerate(anss)}

        dic = {
            'answer': stoi
        }

        with open(path, 'w') as f:
            json.dump(dic, f)

    def dump_vocab_question(self, path=None):
        tokens = []
        max_token_per_question = 0
        for question in self.question_list:
            question = question.lower()[:-1]
            cleaned = nltk.word_tokenize(process_punctuation(question))
            if len(cleaned) > max_token_per_question:
                max_token_per_question = len(cleaned)
            tokens.extend(cleaned)

        word_counter = Counter(tokens).most_common()
        words = [word for word, count in word_counter]

        words = ['<UNK>'] + words
        stoi = {word: i for i, word in enumerate(words)}

        dic = {
            'question': stoi,
            'max_question_length': max_token_per_question
        }

        with open(path, 'w') as f:
            json.dump(dic, f)

    def dump_vocab_ans_500(self, path=None):
        ans_counter = Counter(self.ans_list).most_common()
        anss = [ans for ans, count in ans_counter]

        stoi = {ans: i for i, ans in enumerate(anss[:500])}

        dic = {
            'answer': stoi
        }

        with open(path, 'w') as f:
            json.dump(dic, f)

    def dump_vocab_question_500(self, path=None):
        with open(self.dump_vocab_ans_500_path, 'r') as f:
            dic = json.load(f)
            ans_500_list = dic['answer'].keys()

        question_500_list = []
        for ans in ans_500_list:
            question_500_list.extend(self.ans_question_dict[ans])

        question_500_list = list(set(question_500_list))
        tokens = []
        max_token_per_question = 0
        for question in question_500_list:
            question = question.lower()[:-1]
            cleaned = nltk.word_tokenize(process_punctuation(question))
            if len(cleaned) > max_token_per_question:
                max_token_per_question = len(cleaned)
            tokens.extend(cleaned)

        word_counter = Counter(tokens).most_common()
        words = [word for word, count in word_counter]

        words = ['<UNK>'] + words
        stoi = {word: i for i, word in enumerate(words)}

        dic = {
            'question': stoi,
            'max_question_length': max_token_per_question
        }

        with open(path, 'w') as f:
            json.dump(dic, f)

    def dump_vocab_ans_vocab_relation_500(self, path=None):
        with open(self.dump_vocab_ans_500_path, 'r') as f:
            dic = json.load(f)
            ans_500_list = dic['answer'].keys()

        relation_500_list = []
        for ans in ans_500_list:
            relation_500_list.extend(self.ans_relation_dict[ans])

        relation_500_list = list(set(relation_500_list))
        tokens = []
        for relation in relation_500_list:
            cleaned = nltk.word_tokenize(process_punctuation(relation))
            tokens.extend(cleaned)

        word_counts = Counter(tokens).most_common()
        words = [word for word, count in word_counts]

        words = ['<UNK>'] + words
        stoi = {word: i for i, word in enumerate(words)}

        dic = {
            'answer': stoi
        }

        with open(path, 'w') as f:
            json.dump(dic, f)

if __name__ == '__main__':
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    runner = Runner(cfg)

    runner.get_list('aokvqa_v1p0_train.json')
    runner.dump_all()
    runner.dump_vocab()
