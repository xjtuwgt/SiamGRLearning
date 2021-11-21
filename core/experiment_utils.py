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
    best_accuracy = 0.0
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
            if global_step % args.logging_steps == 0:
                print('Loss at step {} = {:.5f}'.format(global_step, loss.data.item()))
        if (epoch_idx + 1) % 10 == 0:
            eval_acc = evaluate_node_classification_model(model=model, node_data_helper=node_data_helper, args=args)
            if eval_acc > best_accuracy:
                best_accuracy = eval_acc
            print('Best acc = {:.5f}, current acc = {:.5f}'.format(best_accuracy, eval_acc))
    return


def evaluate_node_classification_model(model, node_data_helper, args):
    val_dataloader = node_data_helper.data_loader(data_type='valid')
    logging.info('Loading validation data = {} completed'.format(len(val_dataloader)))
    epoch_iterator = tqdm(val_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    model.eval()
    total_correct = 0.0
    total_example = 0.0
    for step, batch in enumerate(epoch_iterator):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key == 'batch_label':
                batch[key] = value.to(args.device)
            else:
                batch[key] = (value[0].to(args.device), value[1].to(args.device))
        with torch.no_grad():
            logits = model.forward(batch)
            preds = torch.argmax(logits, dim=-1)
            total_example = total_example + preds.shape[0]
            total_correct = total_correct + (preds == batch['batch_label']).sum().data.item()
    eval_acc = total_correct/total_example
    return eval_acc
