from codes.ogb_graph_data import ogb_node_pred_subgraph_data_helper
from codes.citation_graph_data import citation_node_pred_subgraph_data_helper
import logging
from tqdm import tqdm, trange
from codes.gnn_predictor import NodeClassificationModel
import torch
from core.utils import IGNORE_IDX


def train_node_classification(encoder, args):
    # **********************************************************************************
    if args.graph_type == 'citation':
        node_data_helper = citation_node_pred_subgraph_data_helper(args=args)
    elif args.graph_type == 'ogb':
        node_data_helper = ogb_node_pred_subgraph_data_helper(args=args)
    else:
        raise 'Graph type: {} is not supported'.format(args.graph_type)
    logging.info('Number of classes = {}'.format(node_data_helper.num_class))
    train_dataloader = node_data_helper.data_loader(data_type='train')
    logging.info('Loading training data = {} completed'.format(len(train_dataloader)))
    val_dataloader = node_data_helper.data_loader(data_type='valid')
    logging.info('Loading validation data = {} completed'.format(len(val_dataloader)))
    logging.info('*' * 75)
    # **********************************************************************************
    model = NodeClassificationModel(graph_encoder=encoder, encoder_dim=args.siam_dim,
                                    num_of_classes=node_data_helper.num_class, fix_encoder=False)
    model.to(args.device)
    # **********************************************************************************
    loss_fcn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.fine_tuned_learning_rate, weight_decay=args.fine_tuned_weight_decay)
    # **********************************************************************************
    start_epoch = 0
    global_step = 0
    # **********************************************************************************
    logging.info('Starting fine tuning the model...')
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                               disable=args.local_rank not in [-1, 0])
    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            for key, value in batch.items():
                if key == 'batch_label':
                    batch[key] = value.to(args.device)
                else:
                    batch[key] = (value[0].to(args.device), value[1].to(args.device))
            logits = model.forward(batch)
            loss = loss_fcn(logits, batch['batch_label'])
            del batch
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step = global_step + 1

            if global_step % 20 == 0:
                print(loss.data.item())

    return