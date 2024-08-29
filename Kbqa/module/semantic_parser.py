from config import *
import ahocorasick
import pandas as pd
import os
import pickle
from tqdm import tqdm


def build_search_tree(input_folder_path, tree_save_path):
    tree = ahocorasick.Automaton()

    stock_basic = pd.read_csv(os.path.join(input_folder_path, '股票信息.csv'), encoding='gbk')
    for idx, each_row in tqdm(stock_basic.iterrows()):
        tree.add_word(str(each_row['name']), (str(each_row['name']), '股票'))
        tree.add_word(str(each_row['industry']), (str(each_row['industry']), '行业'))

    concept = pd.read_csv(os.path.join(input_folder_path, '概念信息.csv'), encoding='gbk')
    for idx, each_row in tqdm(concept.iterrows()):
        tree.add_word(str(each_row['name']), (str(each_row['name']), '概念'))

    holder = pd.read_csv(os.path.join(input_folder_path, '股东信息.csv'), encoding='gbk')
    for idx, each_row in tqdm(holder.iterrows()):
        tree.add_word(str(each_row['name']), (str(each_row['name']), '股东'))

    tree.make_automaton()

    with open(tree_save_path, 'wb') as fout:
        pickle.dump(tree, fout)


class SemanticParser:
    """实体搜索器"""

    def __init__(self, entity_model_load_path, question_types):
        self.entity_model_load_path = entity_model_load_path
        self.entity_model = self.load_model()
        self.question_types = question_types

    def load_model(self):
        """加载模型"""
        with open(self.entity_model_load_path, 'rb') as fin:
            return pickle.load(fin)

    def predict_question_types(self, query):
        rtn_ques_types = []
        for ques_type, kws in self.question_types.items():
            for each_kw in kws:
                if each_kw in query:
                    rtn_ques_types.append(ques_type)
                    break
        return rtn_ques_types

    def predict(self, query):
        """预测 query"""

        rtn = {}

        ques_types = self.predict_question_types(query)

        entities = {}
        for end_index, (entity_name, entity_type) in self.entity_model.iter(query):
            entities[entity_name] = entity_type

        if len(ques_types) != 0 and len(entities) != 0:
            rtn['ques_types'] = ques_types
            rtn['entities'] = entities

            contexts['ques_types'] = ques_types
            contexts['entities'] = entities

        elif len(ques_types) != 0:
            rtn['ques_types'] = ques_types

            contexts['ques_types'] = ques_types

            rtn['entities'] = contexts['entities']

        elif len(entities) != 0:

            rtn['ques_types'] = contexts['ques_types']

            rtn['entities'] = entities

            contexts['entities'] = entities
        else:

            rtn['ques_types'] = []
            rtn['entities'] = {}

        return rtn


if __name__ == '__main__':
    build_search_tree(entity_corpus_path, entity_searcher_save_path)

