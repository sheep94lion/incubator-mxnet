#ifndef MXNET_OPERATOR_CONTRIB_RPN_INV_NORMALIZE123_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_RPN_INV_NORMALIZE123_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
    namespace op {
        struct RpnInvNormalizeParam: public dmlc::Parameter<RpnInvNormalizeParam> {
            int num_anchors;
            mxnet::Tuple<float> bbox_mean, bbox_std;
            DMLC_DECLARE_PARAMETER(RpnInvNormalizeParam) {
                DMLC_DECLARE_FIELD(num_anchors)
                    .set_default(9)
                    .describe("Num of anchors.");
                DMLC_DECLARE_FIELD(bbox_mean)
                    .set_default({0.0f, 0.0f, 0.0f, 0.0f})
                    .describe("Bbox mean.");
                DMLC_DECLARE_FIELD(bbox_std)
                    .set_default({0.1f, 0.1f, 0.4f, 0.4f})
                    .describe("Bbox std.");
            }
        };

        template<typename xpu>
        void RpnInvNormalizeOpForwardS(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
            mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
            const TBlob& in_data = inputs[0];
            const TBlob& out_data = outputs[0];
            const RpnInvNormalizeParam& param = nnvm::get<RpnInvNormalizeParam>(attrs.parsed);
            using namespace mxnet_op;
            MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
                index_t i1, i2, i3, i4;
                mshadow::Tensor<xpu, 4, DType> in_ts = in_data.get<xpu, 4, DType>(s);
                mshadow::Tensor<xpu, 4, DType> out_ts = out_data.get<xpu, 4, DType>(s);
                for (i1 = 0; i1 < out_ts.size(0); i1++) {
                    for (i2 = 0; i2 < out_ts.size(1); i2++) {
                        for (i3 = 0; i3 < out_ts.size(2); i3++) {
                            for (i4 = 0; i4 < out_ts.size(3); i4++) {
                                float std = param.bbox_std[i2 % 4];
                                float mean = param.bbox_mean[i2 % 4];
                                KERNEL_ASSIGN(out_ts[i1][i2][i3][i4], req[0], in_ts[i1][i2][i3][i4] * std + mean);
                            }
                        }
                    }
                }
            });
        }

        template<int req>
        struct rpn_inv_normalize_forward {
            template<typename DType>
            MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                            const mxnet::Tuple<float> bbox_mean, 
                                            const mxnet::Tuple<float> bbox_std, 
                                            const size_t d1, const size_t d2, const size_t d3) {
                index_t i2 = (i % (d1 * d2 * d3)) / (d2 * d3);

                KERNEL_ASSIGN(out_data[i], req, in_data[i] * bbox_std[i2 % 4] + bbox_mean[i2 % 4]);
            }
        };

        template<typename xpu>
        void RpnInvNormalizeOpForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
            mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
            const TBlob& in_data = inputs[0];
            const TBlob& out_data = outputs[0];
            const RpnInvNormalizeParam& param = nnvm::get<RpnInvNormalizeParam>(attrs.parsed);
            using namespace mxnet_op;
            MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
                MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
                    mshadow::Tensor<xpu, 4, DType> in_ts = in_data.get<xpu, 4, DType>(s);
                    size_t d1 = in_ts.size(1);
                    size_t d2 = in_ts.size(2);
                    size_t d3 = in_ts.size(3);
                    Kernel<rpn_inv_normalize_forward<req_type>, xpu>::Launch(
                        s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
                        param.bbox_mean, param.bbox_std, d1, d2, d3);
                });
            });
        }
    }
}

#endif