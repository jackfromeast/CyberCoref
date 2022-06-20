import re
import os
from tqdm import tqdm
import pandas as pd
import json
from collections import Counter

from config import arg_parse

args = arg_parse()
from dataLoader import load_corpus, Corpus, Document, BERTDocument
import numpy as np

"""
    return entity_info = [(e_index, entity_type, from, to, content), ...]
"""
def read_entity(ann_fname):
    entity_info = []

    for line in open(ann_fname, 'r'):
        # 如果不是T开头 说明已经标了Relation
        if not line.startswith('T'):
            continue
        
        index = re.search(r'^\w\d*', line).group(0)
        e_type = re.search(r'(?<=\t)[\w-]*', line).group(0)
        e_from = re.search(r'(?<=\s)[0-9]{1,}',line).group(0)
        e_to = re.findall(r'(?<=\s)[0-9]{1,}',line)[1]
        e_content = line.split('\t')[-1].strip('\n')

        entity_info.append((index, e_type, e_from, e_to, e_content))

    return entity_info

"""
    读取ann文件中的coref关系
    return : coref_relations = [[T1, T2, ...], [T4, T5, ...]]
"""
def read_coref(ann_fname):
    coref_relations = []

    for line in open(ann_fname, 'r'):
        if not line.startswith('*'):
            continue

        elif re.search(r'(?<=\t)[\w-]*', line).group(0) != 'Coreference':
            continue

        else:
            line = line.rstrip('\n')
            coref_entities = line.split('\t')[1].split(' ')[1:]
            coref_relations.append(coref_entities)

    return coref_relations


def count(rootdir):
    entity_count = {} # 不同类型实体数目
    entity_coref_count = {} # 不同类型实体中存在共指关系的数量
    entity_coref_link_count = {} # 不同类型实体对应的共指链数量
    entity_coref_cluster_count = {} # 不同类型实体对应共指簇的数量
    
    for fpathe,dirs,fs in os.walk(rootdir):
        for file_name in tqdm(fs):
            # 遍历其中所有ann文件
            if not file_name.endswith('.ann'):
                continue
            
            ann_fname =  fpathe + '/' + file_name
            
            entity_info = read_entity(ann_fname)
            corel_info = read_coref(ann_fname)
            
            entity_types = [entity[1] for entity in entity_info]
            for key, value in Counter(entity_types).items():
                if key not in entity_count.keys():
                    entity_count[key] = value
                else:
                    entity_count[key] += value
            
            idx2type = dict(set([(entity[0], entity[1]) for entity in entity_info]))
            
            for cluster in corel_info:
                cluster_entity_types = [idx2type[entity_idx] for entity_idx in cluster]
                
                for key, value in Counter(cluster_entity_types).items():
                    if key not in entity_coref_count.keys():
                        entity_coref_count[key] = value

                        # 对于簇[x,x,y,x,y] entity_coref_link_count[x]=9 entity_coref_link_count[y]=7 
                        link_count_a = value * (value - 1) / 2
                        link_count_b = value * (len(cluster) - value)
                        entity_coref_link_count[key] = int(link_count_a + link_count_b)                                                                                                                                    
                    else:
                        entity_coref_count[key] += value

                        link_count_a = value * (value - 1) / 2
                        link_count_b = value * (len(cluster) - value)
                        entity_coref_link_count[key] += int(link_count_a + link_count_b)
                
                
                cluster_entity_types = set(cluster_entity_types)
                
                for entity_type in set(cluster_entity_types):
                    if entity_type not in entity_coref_cluster_count.keys():
                        entity_coref_cluster_count[key] = 1
                    else:
                        entity_coref_cluster_count[key] += 1
    
    return entity_count, entity_coref_count, entity_coref_link_count, entity_coref_cluster_count



def find_lambda(corpus):
    ratios = []
    for doc in corpus:
        ratios.append(len(doc.corefs)/ len(doc.tokens))
    
    return np.mean(np.array(ratios))


if __name__ == '__main__':
    # entity_count, entity_coref_count, entity_coref_link_count, entity_coref_cluster_count = count('./Dataset/rawData/CasieCoref0418/')
    # with open('./Dataset/dataset_info.json', 'w', encoding='utf-8') as fs:
    #     json.dump([entity_count, entity_coref_count, entity_coref_link_count, entity_coref_cluster_count], fs, indent=4)

    train_corpus_path = args.dataset_path + args.corpus_subpath + '/train' + args.corpus_filename
    val_corpus_path = args.dataset_path + args.corpus_subpath + '/val' + args.corpus_filename
    train_corpus = load_corpus(train_corpus_path)
    val_corpus = load_corpus(val_corpus_path)

    print(find_lambda(train_corpus))
    print(find_lambda(val_corpus))
        
                
            