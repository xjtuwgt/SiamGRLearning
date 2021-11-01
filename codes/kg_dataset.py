from core.kg_utils import KGDataset, kg_data_path_collection, \
    knowledge_graph_construction_from_triples
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
    # for k, v in kg_data.relation2id.items():
    #     print(k, v)
    graph = knowledge_graph_construction_from_triples(num_entities=kg_data.n_entities,
                                                      num_relations=kg_data.n_relations, triples=kg_data.train)
    print(graph)