from codes.ogb_graph_data import ogb_node_pred_subgraph_data_helper
from codes.citation_graph_data import citation_node_pred_subgraph_data_helper
import logging
from codes.gnn_predictor import NodeClassificationModel


def train_node_classification(encoder, args):
    # **********************************************************************************
    if args.graph_type == 'citation':
        node_data_helper = citation_node_pred_subgraph_data_helper(args=args)
    elif args.graph_type == 'ogb':
        node_data_helper = ogb_node_pred_subgraph_data_helper(args=args)
    else:
        raise 'Graph type: {} is not supported'.format(args.graph_type)
    train_dataloader = node_data_helper.data_loader(data_type='train')
    logging.info('Loading training data = {} completed'.format(len(train_dataloader)))
    val_dataloader = node_data_helper.data_loader(data_type='valid')
    logging.info('Loading validation data = {} completed'.format(len(val_dataloader)))
    logging.info('*' * 75)
    # **********************************************************************************
    # model = NodeClassificationModel(graph_encoder=encoder, encoder_dim=)
    return