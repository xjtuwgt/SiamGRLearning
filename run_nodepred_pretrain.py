import logging
import torch
from torch import nn
from tqdm import tqdm, trange
import sys
from tensorboardX import SummaryWriter
from codes.argument_parser import default_parser, json_to_argv, complete_default_parser
from codes.citation_graph_data import citation_subgraph_pretrain_dataloader
from codes.ogb_graph_data import ogb_subgraph_pretrain_dataloader
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
if args.graph_type == 'citation':
    pretrain_dataloader, node_features, n_classes = citation_subgraph_pretrain_dataloader(args=args)
    logging.info('Loading pretrained data = {} for {} completed'.format(len(pretrain_dataloader), args.graph_type))
    logging.info('*' * 75)
elif args.graph_type == 'ogb':
    pretrain_dataloader, node_features, n_classes = ogb_subgraph_pretrain_dataloader(args=args)
    logging.info('Loading pretrained data = {} for {} completed'.format(len(pretrain_dataloader), args.graph_type))
    logging.info('*' * 75)
else:
    raise 'Graph type = {} is not supported'.format(args.graph_type)
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
    t_total_pretrain_steps = args.total_pretrain_steps
    args.num_pretrain_epochs = args.total_pretrain_steps // (len(pretrain_dataloader)
                                                             // args.gradient_accumulation_steps) + 1
else:
    t_total_pretrain_steps = len(pretrain_dataloader) // args.gradient_accumulation_steps \
                    * args.num_pretrain_epochs
optimizer, scheduler = graph_encoder.pretrain_optimizer_scheduler(total_steps=t_total_pretrain_steps)
# #########################################################################
global_step = 0
if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)
graph_encoder.zero_grad()
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
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if args.graph_type == 'citation':
    logging.info('Staring Pretraining over graph = {}'.format(args.citation_name))
elif args.graph_type == 'ogb':
    logging.info('Staring Pretraining over graph = {}'.format(args.ogb_node_name))
else:
    raise 'Graph type {} is not supported'.format(args.graph_type)
pretrain_iterator = trange(start_epoch, start_epoch+int(args.num_pretrain_epochs), desc="Epoch",
                           disable=args.local_rank not in [-1, 0])
for epoch in pretrain_iterator:
    epoch_iterator = tqdm(pretrain_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        graph_encoder.train()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            batch[key] = (value[0].to(args.device), value[1].to(args.device))
        p1, p2, z1, z2 = graph_encoder.forward(batch)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5 + 1
        del batch
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), args.max_grad_norm)
        training_logs.append({'Train_loss': loss.data.item()})
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            graph_encoder.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                training_logs = []
                logging.info('Pre-trained model evaluation at step_{}/epoch_{}'.format(global_step, epoch + 1))
                for key, value in metrics.items():
                    logging.info('Metric {}: {:.5f}'.format(key, value))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
