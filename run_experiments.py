import logging
import torch
import sys
from codes.argument_parser import default_parser, json_to_argv, complete_default_parser
from codes.gnn_encoder import GDTEncoder

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
for key, value in vars(args).items():
    logger.info('Hype-parameter\t{} = {}'.format(key, value))
logging.info('*' * 75)
#########################################################################
graph_encoder = GDTEncoder(config=args)
graph_encoder.to(args.device)
# #########################################################################
# # Show model information
# #########################################################################
logging.info('Model Parameter Configuration:')
for name, param in graph_encoder.named_parameters():
    logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
logging.info('*' * 75)