//#include "./quadratic_op-inl.h"
#include "./rpn_inv_normalize_op-inl.h"

namespace mxnet {
    namespace op {
        DMLC_REGISTER_PARAMETER(RpnInvNormalizeParam);

        NNVM_REGISTER_OP(_contrib_rpn_inv_normalize)
        .describe("This is my first test operator. I hope it will succeed soon!")
        .set_attr_parser(ParamParser<RpnInvNormalizeParam>)
        .set_num_inputs(1)
        .set_num_outputs(1)
        .set_attr<nnvm::FListInputNames>("FListInputNames",
          [](const NodeAttrs& attrs) {
              return std::vector<std::string>{"data"};
          })
        .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
        .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
        .set_attr<FCompute>("FCompute<cpu>", RpnInvNormalizeOpForward<cpu>);
    }
}