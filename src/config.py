"""
main.py
Autor: JungwooChoi, HyeongwonKang
Incremental pseudo labeling for anomaly detection Argument Parser

"""
import argparse


def load_config():
    """
    argument parser

    """
    ap = argparse.ArgumentParser()

    ap.add_argument("-M", "--model", type=str, required=True, help="model(StackedGRU, AE, VAE)")

    args = vars(ap.parse_args())

    return args

def str2bool(v):
    if type(v) is not bool:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    else:
        return v


