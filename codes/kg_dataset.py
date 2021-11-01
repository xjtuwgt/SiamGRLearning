from evens import KG_DATA_FOLDER
from core.kg_utils import KGDataset, kg_data_path_collection, \
    knowledge_graph_construction_from_triples
from core.graph_utils import construct_special_graph_dictionary

def knowledge_graph_khop_reconstruction(dataset: str, hop_num=5, bi_directed:bool=True):
    print('Bi-directional homogeneous graph: {}'.format(dataset))
    entity_path, relation_path, train_path, _, _ = kg_data_path_collection(kg_path=KG_DATA_FOLDER, kg_name=dataset)
    kg_data = KGDataset(entity_path=entity_path, relation_path=relation_path, train_path=train_path)
    graph = knowledge_graph_construction_from_triples(num_entities=kg_data.n_entities,
                                                      num_relations=kg_data.n_relations,
                                                      triples=kg_data.train, bi_directional=bi_directed)
    nentities = kg_data.n_entities
    nrelations = 2 * kg_data.n_relations if bi_directed else kg_data.n_relations
    graph, number_of_nodes, number_of_relations, \
    special_entity_dict, special_relation_dict = construct_special_graph_dictionary(graph=graph, n_entities=nentities,
                                       n_relations=nrelations, hop_num=hop_num)
    number_of_added_nodes = number_of_nodes - nentities
    print('Added number of nodes = {}'.format(number_of_added_nodes))
    assert len(special_entity_dict) == number_of_added_nodes
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict