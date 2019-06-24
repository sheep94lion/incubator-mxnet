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
    }
}
// namespace mxnet {
// namespace op {

// struct QuadraticParam : public dmlc::Parameter<QuadraticParam> {
//   float a, b, c;
//   DMLC_DECLARE_PARAMETER(QuadraticParam) {
//     DMLC_DECLARE_FIELD(a)
//       .set_default(0.0)
//       .describe("Coefficient of the quadratic term in the quadratic function.");
//     DMLC_DECLARE_FIELD(b)
//       .set_default(0.0)
//       .describe("Coefficient of the linear term in the quadratic function.");
//     DMLC_DECLARE_FIELD(c)
//       .set_default(0.0)
//       .describe("Constant term in the quadratic function.");
//   }
// };

// struct RpnInvNormalizeParam: public dmlc::Parameter<RpnInvNormalizeParam> {
//     int num_anchors;
//     float bbox_mean[4], bbox_std[4];
//     DMLC_DECLARE_PARAMETER(RpnInvNormalizeParam) {
//         DMLC_DECLARE_FIELD(num_anchors)
//             .describe("Num of anchors.");
//         DMLC_DECLARE_FIELD(bbox_mean)
//             .describe("Bbox mean.");
//         DMLC_DECLARE_FIELD(bbox_std)
//             .describe("Bbox std.");
//     }
// };

// template<int req>
// struct quadratic_forward {
//   template<typename DType>
//   MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
//                                   const float a, const float b, const float c) {
//     KERNEL_ASSIGN(out_data[i], req, in_data[i] * (a * in_data[i] + b) + c);
//   }
// };

// template<int req>
// struct quadratic_backward {
//   template<typename DType>
//   MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
//                                   const DType* in_data, const float a, const float b) {
//     KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * (2 * a * in_data[i] + b));
//   }
// };

// template<typename xpu>
// void QuadraticOpForward(const nnvm::NodeAttrs& attrs,
//                         const OpContext& ctx,
//                         const std::vector<TBlob>& inputs,
//                         const std::vector<OpReqType>& req,
//                         const std::vector<TBlob>& outputs) {
//   CHECK_EQ(inputs.size(), 1U);
//   CHECK_EQ(outputs.size(), 1U);
//   CHECK_EQ(req.size(), 1U);
//   mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
//   const TBlob& in_data = inputs[0];
//   const TBlob& out_data = outputs[0];
//   const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);
//   using namespace mxnet_op;
//   MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
//     MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
//       Kernel<quadratic_forward<req_type>, xpu>::Launch(
//           s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
//           param.a, param.b, param.c);
//     });
//   });
// }

// template<typename xpu>
// void QuadraticOpForwardCsrImpl(const QuadraticParam& param,
//                                const OpContext& ctx,
//                                const NDArray& input,
//                                const OpReqType req,
//                                const NDArray& output) {
//   using namespace mshadow;
//   using namespace mxnet_op;
//   using namespace csr;
//   if (req == kNullOp) return;
//   CHECK_EQ(req, kWriteTo) << "QuadraticOp with CSR only supports kWriteTo";
//   Stream<xpu> *s = ctx.get_stream<xpu>();
//   if (!input.storage_initialized()) {
//     FillZerosCsrImpl(s, output);
//     return;
//   }
//   const nnvm::dim_t nnz = input.storage_shape()[0];
//   const nnvm::dim_t num_rows = output.shape()[0];
//   output.CheckAndAlloc({Shape1(num_rows + 1), Shape1(nnz)});
//   CHECK_EQ(output.aux_type(kIdx), output.aux_type(kIndPtr))
//     << "The dtypes of indices and indptr don't match";
//   MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
//     MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), IType, {
//       MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
//         Kernel<quadratic_forward<req_type>, xpu>::Launch(
//             s, nnz, output.data().dptr<DType>(), input.data().dptr<DType>(),
//             param.a, param.b, param.c);
//         Copy(output.aux_data(kIdx).FlatTo1D<xpu, IType>(s),
//              input.aux_data(kIdx).FlatTo1D<xpu, IType>(s), s);
//         Copy(output.aux_data(kIndPtr).FlatTo1D<xpu, IType>(s),
//              input.aux_data(kIndPtr).FlatTo1D<xpu, IType>(s), s);
//       });
//     });
//   });
// }

// template<typename xpu>
// void QuadraticOpForwardEx(const nnvm::NodeAttrs& attrs,
//                           const OpContext& ctx,
//                           const std::vector<NDArray>& inputs,
//                           const std::vector<OpReqType>& req,
//                           const std::vector<NDArray>& outputs) {
//   CHECK_EQ(inputs.size(), 1U);
//   CHECK_EQ(outputs.size(), 1U);
//   CHECK_EQ(req.size(), 1U);
//   const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);
//   const auto in_stype = inputs[0].storage_type();
//   const auto out_stype = outputs[0].storage_type();
//   if (in_stype == kCSRStorage && out_stype == kCSRStorage && param.c == 0.0) {
//     QuadraticOpForwardCsrImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0]);
//   } else {
//     LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
//   }
// }

// template<typename xpu>
// void QuadraticOpBackward(const nnvm::NodeAttrs& attrs,
//                          const OpContext& ctx,
//                          const std::vector<TBlob>& inputs,
//                          const std::vector<OpReqType>& req,
//                          const std::vector<TBlob>& outputs) {
//   CHECK_EQ(inputs.size(), 2U);
//   CHECK_EQ(outputs.size(), 1U);
//   CHECK_EQ(req.size(), 1U);
//   mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
//   const TBlob& out_grad = inputs[0];
//   const TBlob& in_data = inputs[1];
//   const TBlob& in_grad = outputs[0];
//   const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);
//   using namespace mxnet_op;
//   MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
//     MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
//       Kernel<quadratic_backward<req_type>, xpu>::Launch(
//           s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
//           in_data.dptr<DType>(), param.a, param.b);
//     });
//   });
// }

    // }  // namespace op
// }  // namespace mxnet

#endif