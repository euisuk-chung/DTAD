"""
main.py
Autor: JungwooChoi, HyeongwonKang
Incremental pseudo labeling for anomaly detection Argument Parser

"""
import argparse

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

def load_config():
    """
    argument parser

    """
    ap = argparse.ArgumentParser()

    ap.add_argument("-M", "--model", type=str, required=True, help="model(StackedGRU, AE, VAE)")
    ap.add_argument("-T", "--type", type=str, default='Train', help="Train or Test")
    ap.add_argument("-G", "--gpu", type=int, default=0, help="gpu number")
    ap.add_argument("-R", "--range_check", default=30, help="range check", type=int)
    ap.add_argument("-H", "--threshold", type=float, default=0.025, help="anomaly score threshold")


    args = vars(ap.parse_args())

    return args


