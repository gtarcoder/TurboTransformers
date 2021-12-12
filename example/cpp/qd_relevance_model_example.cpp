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

#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <string>
#include <thread>

#include "qd_relevance_model.h"
#include "turbo_transformers/core/allocator/allocator_api.h"
#include "turbo_transformers/core/config.h"

static bool test_bert(const std::string &model_path, bool use_cuda = false) {
  // construct a bert model using n_layers and n_heads,
  // the hidden_size can be infered from the parameters

  QDRelevanceModel model(model_path,
                         use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
                         1, /* n_layers */
                         1 /* *n_heads */);
  // std::string query_input = "冰雪奇缘";
  // std::string doc_input =
  //     std::string("冰雪奇缘的番外短片，原来是这么来的！") + "_" +
  //     "娱乐你最懂_冰雪奇缘：生日惊喜;动画电影;几分钟看电影;美国电影";

  std::vector<std::vector<int64_t>> input_ids{
      {101,  1102, 7434, 1936, 5357, 102,  1102, 7434, 1936, 5357, 4638,
       4528, 1912, 4764, 4275, 8024, 1333, 3341, 3221, 6821, 720,  3341,
       4638, 8013, 142,  2031, 727,  872,  3297, 2743, 142,  1102, 7434,
       1936, 5357, 8038, 4495, 3189, 2661, 1599, 132,  1220, 4514, 4510,
       2512, 132,  1126, 1146, 7164, 4692, 4510, 2512, 132,  5401, 1744,
       4510, 2512, 102,  0,    0,    0,    0,    0,    0}};

  std::vector<std::vector<int64_t>> input_masks{
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}};

  std::vector<std::vector<int64_t>> segment_ids{
      {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0}};

  auto vec = model(input_ids, input_masks, segment_ids, PoolType::kFirst);
  std::cout << "#### result: " << std::endl;
  for (auto &v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
  return true;
}

static bool test_memory_opt_bert(const std::string &model_path,
                                 bool use_cuda = false) {
  // construct a bert model using n_layers and n_heads,
  // the hidden_size can be infered from the parameters

  auto &allocator =
      turbo_transformers::core::allocator::Allocator::GetInstance();
  allocator.set_config({2, 4, 12, 768, 12});

  QDRelevanceModel model(model_path,
                         use_cuda ? DLDeviceType::kDLGPU : DLDeviceType::kDLCPU,
                         1, /* n_layers */
                         1 /* *n_heads */);
  std::vector<std::vector<int64_t>> input_ids{{12166, 10699, 16752, 4454},
                                              {5342, 16471, 817, 16022}};
  std::vector<std::vector<int64_t>> input_masks{{1, 0, 0, 0}, {1, 1, 1, 0}};
  std::vector<std::vector<int64_t>> segment_ids{{1, 1, 1, 0}, {1, 0, 0, 0}};
  auto vec = model(input_ids, input_masks, segment_ids, PoolType::kFirst);
  std::cout << "#### result: " << std::endl;
  for (auto &v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
  return true;
}

static std::vector<float> CallBackFunction(
    const std::shared_ptr<QDRelevanceModel> model,
    const std::vector<std::vector<int64_t>> input_ids,
    const std::vector<std::vector<int64_t>> position_ids,
    const std::vector<std::vector<int64_t>> segment_ids, PoolType pooltype) {
  return model->operator()(input_ids, position_ids, segment_ids, pooltype);
}

using namespace turbo_transformers;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "./qd_relevance_model_example npz_model_path" << std::endl;
    return -1;
  }
  const std::string model_path = static_cast<std::string>(argv[1]);

  std::cout << "run bert on CPU, use 4 threads to do bert inference"
            << std::endl;
  turbo_transformers::core::SetNumThreads(4);
  test_bert(model_path, false /*not use cuda*/);
  // turbo_transformers::core::SetNumThreads(1);
  // if (core::IsCompiledWithCUDA()) {
  //   std::cout << "10 threads do 10 independent bert inferences." <<
  //   std::endl; test_multiple_threads(model_path, false /*only_input*/,
  //   true
  //   /*use cuda*/,
  //                         10);
  // }

  // test_multiple_threads(model_path, false /*only_input*/,
  //                       false /*not use cuda*/, 1);

  return 0;
}
