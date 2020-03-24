// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo_transformers/layers/bert_pooler.h"

#include <loguru.hpp>

#include "turbo_transformers/core/aligned_scratchpad.h"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"

namespace turbo_transformers {
namespace layers {

void BertPooler::operator()(const core::Tensor& input_tensor,
                                  core::Tensor* output_tensor) const {
  turbo_transformers::core::Tensor temp(
      turbo_transformers::core::NewDLPackTensorT<float>({1, input_tensor.shape(1), input_tensor.shape (2)}));
  auto* data = temp.mutableData<float>();
  auto* input = input_tensor.data<float>();
  for (int i = 0; i < temp.cols (); ++i)
    data[i] = input[i];
  output_tensor->Reshape<float>(
      {1, input_tensor.shape(1), dense_weight_.shape(0)},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::MatMul(temp, false, dense_weight_, true, 1.0, output_tensor,
                  0.0);
  kernels::AddBiasGeLUAct<float>(dense_bias_, output_tensor);
}

void BertPooler::EnforceShapeAndType() const {
  FT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  FT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  FT_ENFORCE_EQ(dense_weight_.shape(0), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(0),
                dense_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
      std::ostringstream os;
      os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
      dense_weight_.Print<float>(os);
      os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
      dense_bias_.Print<float>(os);
      LOG_S(3) << os.str();
    }
}

}  // namespace layers
}  // namespace turbo_transformers
