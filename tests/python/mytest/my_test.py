import mxnet as mx
from operator_py.rpn_inv_normalize import *

data = mx.symbol.Variable('data')
rpn = mx.sym.Custom(
    bbox_pred=data, 
    op_type="rpn_inv_normalize", 
    num_anchors=9,
    bbox_mean=[0.0, 0, 0.0, 0.0],
    bbox_std=[0.1, 0.1, 0.4, 0.4]
)
    
