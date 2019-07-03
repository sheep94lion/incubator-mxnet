#%%
import mxnet as mx
from operator_py.rpn_inv_normalize import *

#%%
data = mx.symbol.Variable('data')
rpn = mx.sym.Custom(
    bbox_pred=data, 
    op_type="rpn_inv_normalize", 
    num_anchors=9,
    bbox_mean=[0.0, 0, 0.0, 0.0],
    bbox_std=[0.1, 0.1, 0.4, 0.4]
)
rpn_output = mx.nd.load("/home/yizhao/Code/mxnet-dev/tests/python/mytest/rpn_bbox_pred_output_dict")


#%%
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

ctx = mx.cpu()
mod = mx.mod.Module(symbol=rpn, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 36, 36, 63))])
mod.init_params()
mod.forward(Batch([rpn_output['rpn_bbox_pred_output']]))

#%%
output1 = mod.get_outputs()[0].asnumpy()
data2 = mx.symbol.Variable('data')
rpn2 = mx.sym.contrib.rpn_inv_normalize(data=data2, num_anchors=9, bbox_mean=[0.0, 0.0, 0.0, 0.0], bbox_std=[0.1, 0.1, 0.4, 0.4])
mod2 = mx.mod.Module(symbol=rpn2, context=ctx, label_names=None)
mod2.bind(for_training=False, data_shapes=[('data', (1, 36, 36, 63))])
mod2.init_params()
mod2.forward(Batch([rpn_output['rpn_bbox_pred_output']]))

#%%
output2 = mod2.get_outputs()[0].asnumpy()

#%%
print output1.shape
print output2.shape

#%%
import numpy as np
print np.array_equal(output1, output2)

#%%
