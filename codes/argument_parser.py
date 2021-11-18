import json
import os
import argparse
import torch
from os.path import join
from core.utils import seed_everything
from core.gpu_utils import get_single_free_gpu
from evens import HOME_DATA_FOLDER, OUTPUT_FOLDER, KG_DATA_FOLDER

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def json_to_argv(json_file):
    j = json.load(open(json_file))
    argv = []
    for k, v in j.items():
        new_v = str(v) if v is not None else None
        argv.extend(['--' + k, new_v])
    return argv

def set_seed(args):
    ##+++++++++++++++++++++++
    random_seed = args.seed + args.local_rank
    ##+++++++++++++++++++++++
    seed_everything(seed=random_seed)

def complete_default_parser(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # set n_gpu
    #+++++++++++++++
    if HOME_DATA_FOLDER.startswith('/dfs/scratch0'):
        args.stanford = 'true'
    if args.local_rank == -1:
        if args.stanford:
            if torch.cuda.is_available():
                gpu_idx, _ = get_single_free_gpu()
                device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # output dir name
    if not args.exp_name:
        args.exp_name = '_'.join(['lr.' + str(args.learning_rate), 'bs.' + str(args.train_batch_size)])
    args.exp_name = os.path.join(args.output_dir, args.exp_name)
    set_seed(args)
    os.makedirs(args.exp_name, exist_ok=True)
    torch.save(args, join(args.exp_name, "training_args.bin"))
    return args

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None,
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--config_file",
                        type=str,
                        default=None,
                        help="configuration file for command parser")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    parser.add_argument('--data_path', type=str, default=HOME_DATA_FOLDER)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--kg_data_path', type=str, default=KG_DATA_FOLDER)
    parser.add_argument('--kg_name', type=str, default='FB15k-237')
    parser.add_argument('--kg_bi_directed', type=boolean_string, default='true')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--citation_name', type=str, default='citeseer')
    parser.add_argument('--ogb_node_name', type=str, default='ogbn-arxiv')
    parser.add_argument('--graph_type', type=str, default='ogb', choices=["citation", "ogb"])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--sub_graph_fanouts', type=str, default='10,5,5,5')
    parser.add_argument('--sub_graph_hop_num', type=int, default=6)
    parser.add_argument('--sub_graph_edge_dir', type=str, default='in')
    parser.add_argument('--arw_position', type=boolean_string, default='true')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--fix_pred_lr', type=boolean_string, default='false')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--relation_number', type=int, default=2)
    parser.add_argument('--relation_emb_dim', type=int, default=300)
    parser.add_argument('--node_number', type=int, default=300)
    parser.add_argument('--node_emb_dim', type=int, default=300)
    parser.add_argument('--arw_pos_emb_dim', type=int, default=300)
    parser.add_argument('--siam_dim', type=int, default=1024)
    parser.add_argument('--siam_pred_dim', type=int, default=512)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--feat_drop', type=float, default=0.25)
    parser.add_argument('--attn_drop', type=float, default=0.25)
    parser.add_argument('--residual', type=boolean_string, default='true')
    parser.add_argument('--diff_head_tail', type=boolean_string, default='false')
    parser.add_argument('--ppr_diff', type=boolean_string, default='true')
    parser.add_argument('--stanford', type=boolean_string, default='true')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--gnn_hop_num', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--negative_slope', type=float, default=1.0)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=16)
    # Environment+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--cpu_num', type=int, default=8)
    parser.add_argument("--data_parallel", default='false', type=boolean_string, help="use data parallel or not")
    parser.add_argument("--gpu_id", default=None, type=str, help="GPU id")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # learning and log ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--seed', type=int, default=43, help="random seed for initialization")
    parser.add_argument("--num_pretrain_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--total_pretrain_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_interval_ratio', type=float, default=0.25,
                        help="evaluate every X updates steps.")

    parser.add_argument("--optimizer", type=str, default="Adam", choices=["AdamW", "Adam"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restart"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--debug", type=boolean_string, default='false')
    return parser
