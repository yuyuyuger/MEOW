#import h5py
import pickle
import torch
import numpy as np

def read_entity_from_id(path):
    entity2id = {}
    with open(path + 'entity2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            entity2id[instance[0]] = int(instance[1])

    return entity2id


def read_relation_from_id(path):
    relation2id = {}
    with open(path + 'relation2id.txt', 'r') as f:
        for line in f:
            instance = line.strip().split()
            relation2id[instance[0]] = int(instance[1])

    return relation2id

def get_adj(path, split):
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)
    triples = []
    rows, cols, data = [], [], []
    unique_entities = set()
    with open(path+split+'.txt', 'r') as f:
        for line in f:
            instance = line.strip().split("\t")
            e1, r, e2 = instance[0], instance[2], instance[1]
            # 检查每个实体和关系是否存在于相应的字典中
            if e1 not in entity2id:
                print(f"Warning: Entity '{e1}' not found in entity2id.")
                continue
            if e2 not in entity2id:
                print(f"Warning: Entity '{e2}' not found in entity2id.")
                continue
            if r not in relation2id:
                print(f"Warning: Relation '{r}' not found in relation2id.")
                continue
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples.append((entity2id[e1], relation2id[r], entity2id[e2]))
            rows.append(entity2id[e2])
            cols.append(entity2id[e1])
            data.append(relation2id[r])

        return triples, (rows, cols, data), unique_entities
# Calculate adjacency matrix

def load_data(datasets):
    path = '../data/' + datasets + '/_data_/'
    path3 = '../data/' + datasets + '/embeddings/'
    train_triples, train_adj, train_unique_entities = get_adj(path, 'train')
    val_triples, val_adj, val_unique_entities = get_adj(path, 'valid')
    test_triples, test_adj, test_unique_entities = get_adj(path, 'test')
    entity2id = read_entity_from_id(path)
    relation2id = read_relation_from_id(path)


    img_features = pickle.load(open(path3 + 'img_features.pkl', 'rb'))
    text_features = pickle.load(open(path3+'text_features.pkl', 'rb'))

    return entity2id, relation2id, img_features, text_features, \
           (train_triples, train_adj, train_unique_entities), \
           (val_triples, val_adj, val_unique_entities), \
           (test_triples, test_adj, test_unique_entities)





