import torch
import argparse
import yaml
from semilearn.core.utils import get_net_builder
from semilearn.imb_algorithms import get_imb_algorithm

config = yaml.load(open('config/_linux_ratio_sweep/001-fixmatch_metaexpert_pic_lb900_50_ulb1800_50_0.0_2_linux_eval.yaml'), Loader=yaml.Loader)
args = argparse.Namespace(**config)
args.imb_algorithm = "metaexpert"
args.num_classes = 9
args.distributed = False
args.use_pretrain = False
args.pretrain_path = None
# Build net
net_builder = get_net_builder(args.net, from_name=False)
net = get_imb_algorithm(args, net_builder, None, None, None)

m = None
if hasattr(net, 'model'):
    m = net.model
print("Model attributes:", [k for k in dir(m) if 'tau' in k or 'hat' in k])
if hasattr(m, 'module'):
    print("Module attributes:", [k for k in dir(m.module) if 'tau' in k or 'hat' in k])
