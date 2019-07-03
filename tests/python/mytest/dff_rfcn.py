#%%
import argparse
import os
import glob
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np

#%%
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/dff_rfcn/cfgs/dff_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes, draw_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deep Feature Flow demo')
    args = parser.parse_args()
    return args

args = parse_args()

#%%
# get symbol
pprint.pprint(config)
config.symbol = 'resnet_v1_101_flownet_rfcn'
model = '/../model/rfcn_dff_flownet_vid'
sym_instance = eval(config.symbol + '.' + config.symbol)()
key_sym = sym_instance.get_key_test_symbol(config)
cur_sym = sym_instance.get_cur_test_symbol(config)