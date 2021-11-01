import numpy as np
from os.path import join

def kg_data_path_collection(kg_path, kg_name: str):
    entity_path = join(kg_path, kg_name, 'entities.dict')
    relation_path = join(kg_path, kg_name, 'relations.dict')
    train_path = join(kg_path, kg_name, 'train.txt')
    valid_path = join(kg_path, kg_name, 'valid.txt')
    test_path = join(kg_path, kg_name, 'test.txt')
    return entity_path, relation_path, train_path, valid_path, test_path

class KGDataset(object):
    '''Load a knowledge graph
    The folder with a knowledge graph has five files:
    * entities stores the mapping between entity Id and entity name.
    * relations stores the mapping between relation Id and relation name.
    * train stores the triples in the training set.
    * valid stores the triples in the validation set.
    * test stores the triples in the test set.
    The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.
    The triples are stored as 'head_name\trelation_name\ttail_name'.
    '''
    def __init__(self, entity_path, relation_path, train_path,
                 valid_path=None, test_path=None, format=(0,1,2),
                 delimiter='\t', skip_first_line=False):
        self.delimiter = delimiter
        self.entity2id, self.n_entities = self.read_entity(entity_path)
        self.relation2id, self.n_relations = self.read_relation(relation_path)
        self.train = self.read_triple(train_path, "train", skip_first_line, format)
        if valid_path is not None:
            self.valid = self.read_triple(valid_path, "valid", skip_first_line, format)
        else:
            self.valid = None
        if test_path is not None:
            self.test = self.read_triple(test_path, "test", skip_first_line, format)
        else:
            self.test = None

    def read_entity(self, entity_path):
        with open(entity_path) as f:
            entity2id = {}
            for line in f:
                eid, entity = line.strip().split(self.delimiter)
                entity2id[entity] = int(eid)

        return entity2id, len(entity2id)

    def read_relation(self, relation_path):
        with open(relation_path) as f:
            relation2id = {}
            for line in f:
                rid, relation = line.strip().split(self.delimiter)
                relation2id[relation] = int(rid)

        return relation2id, len(relation2id)

    def read_triple(self, path, mode, skip_first_line=False, format=(0,1,2)):
        # mode: train/valid/test
        if path is None:
            return None

        print('Reading {} triples....'.format(mode))
        heads = []
        tails = []
        rels = []
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split(self.delimiter)
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                heads.append(self.entity2id[h])
                rels.append(self.relation2id[r])
                tails.append(self.entity2id[t])

        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))
        return (heads, rels, tails)

if __name__ == '__main__':
    import os
    from evens import KG_DATA_FOLDER
    kg_name = 'wn18rr'
    entity_path, relation_path, train_path, _, _ = kg_data_path_collection(kg_path=KG_DATA_FOLDER, kg_name=kg_name)
    kg_data = KGDataset(entity_path=entity_path, relation_path=relation_path,
                        train_path=train_path)
    # train_data = kg_data.train
    # print(train_data)
    # print(kg_data.entity2id)
    for k, v in kg_data.relation2id.items():
        print(k, v)