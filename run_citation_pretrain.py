import logging
import torch
from torch import nn
from tqdm import tqdm, trange
import sys
from codes.argument_parser import default_parser, json_to_argv, complete_default_parser
from codes.citation_graph_data import citation_subgraph_pretrain_dataloader
from codes.citation_graph_data import citation_subgraph_data_helper
from codes.gnn_encoder import GraphSimSiamEncoder
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# #######################################################################
# # Initialize arguments
# #######################################################################
parser = default_parser()
logger.info("IN CMD MODE")
logger.info("Pytorch version = {}".format(torch.__version__))
# #######################################################################
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
# #######################################################################
args = complete_default_parser(args)
#########################################################################
logging.info('*' * 75)
for key, value in vars(args).items():
    logger.info('Hype-parameter\t{} = {}'.format(key, value))
logging.info('*' * 75)
#########################################################################
citation_pretrain_dataloader, node_features, n_classes = citation_subgraph_pretrain_dataloader(args=args)
logging.info('Loading pretrained data = {} completed'.format(len(citation_pretrain_dataloader)))
logging.info('*' * 75)
citation_data_helper = citation_subgraph_data_helper(args=args)
citation_train_dataloader = citation_data_helper.data_loader(data_type='train')
logging.info('Loading training data = {} completed'.format(len(citation_train_dataloader)))
citation_val_dataloader = citation_data_helper.data_loader(data_type='validation')
logging.info('Loading validation data = {} completed'.format(len(citation_val_dataloader)))
logging.info('*' * 75)
#########################################################################
for key, value in vars(args).items():
    if 'number' in key or 'emb_dim' in key:
        logger.info('Hype-parameter\t{} = {}'.format(key, value))
logging.info('*' * 75)
#########################################################################
graph_encoder = GraphSimSiamEncoder(config=args)
graph_encoder.init(graph_node_emb=node_features)
graph_encoder.to(args.device)
# #########################################################################
# # Print model information
# #########################################################################
if args.total_pretrain_steps > 0:
    t_total_steps = args.total_pretrain_steps
    args.num_pretrain_epochs = args.total_pretrain_steps // (len(citation_pretrain_dataloader)
                                                             // args.gradient_accumulation_steps) + 1
else:
    t_total_steps = len(citation_pretrain_dataloader) // args.gradient_accumulation_steps \
                    * args.num_pretrain_epochs
optimizer, scheduler = graph_encoder.pretrain_optimizer_scheduler(total_steps=t_total_steps)
# #########################################################################
logging.info('Model Parameter Configuration:')
for name, param in graph_encoder.named_parameters():
    logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
logging.info('*' * 75)
# #########################################################################
criterion = nn.CosineSimilarity(dim=1)
# #########################################################################
start_epoch = 0
best_accuracy = 0.0
best_model_name = None
training_logs = []
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pretrain_iterator = trange(start_epoch, start_epoch+int(args.num_pretrain_epochs), desc="Epoch",
                        disable=args.local_rank not in [-1, 0])
for epoch_idx, epoch in enumerate(pretrain_iterator):
    epoch_iterator = tqdm(citation_pretrain_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        graph_encoder.train()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            batch[key] = (value[0].to(args.device), value[1].to(args.device))
        p1, p2, z1, z2 = graph_encoder.forward(batch)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5 + 1
        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), args.max_grad_norm)
        training_logs.append({'loss': loss.data.item()})
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            graph_encoder.zero_grad()

        print(loss)