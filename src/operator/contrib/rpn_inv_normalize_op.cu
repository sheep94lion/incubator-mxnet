#include "./rpn_inv_normalize_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(rpn_inv_normalize)
.set_attr<FCompute>("FCompute<gpu>", RpnInvNormalizeOpForward<gpu>);

}  // namespace op
}  // namespace mxnet