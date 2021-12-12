// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "turbo_transformers/layers/fully_connected.h"

#include <loguru.hpp>

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {

void FullyConnected::operator()(const core::Tensor &input_tensor,
                                core::Tensor *output_tensor) const {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("FullyConnected", input_tensor.device_type());
#endif

  output_tensor->Reshape<float>({input_tensor.shape(0), dense_weight_.shape(1)},
                                input_tensor.device_type(),
                                input_tensor.device_id(), name_);
  kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,
                  0.0, "FullyConnected/MatMul");
  switch (act_type_) {
    case types::ActivationType::None:
      kernels::AddBiasAct<float, types::ActivationType::None>(
          dense_bias_, output_tensor, "AddBiasAct");
      break;
    case types::ActivationType::Gelu:
      kernels::AddBiasAct<float, types::ActivationType::Gelu>(
          dense_bias_, output_tensor, "AddBiasAct");
      break;
    case types::ActivationType::Tanh:
      kernels::AddBiasAct<float, types::ActivationType::Tanh>(
          dense_bias_, output_tensor, "AddBiasAct");
      break;
    case types::ActivationType::Relu:
      kernels::AddBiasAct<float, types::ActivationType::Relu>(
          dense_bias_, output_tensor, "AddBiasAct");
      break;
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("FullyConnected", input_tensor.device_type());
#endif
}

void FullyConnected::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::stringstream ss;
    ss << "<<<<<<<< dense_weight_ <<<<<<<<<<";
    dense_weight_.Print<float>(ss);
    ss << "<<<<<<<< dense_bias <<<<<<<<<<";
    dense_bias_.Print<float>(ss);
    LOG_S(3) << ss.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers
