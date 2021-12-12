#include "qd_relevance_model.h"

#include <string>
#include <utility>

#include "cnpy.h"
#include "turbo_transformers/core/tensor_copy.h"
#include "turbo_transformers/layers/bert_attention.h"
#include "turbo_transformers/layers/bert_embedding.h"
#include "turbo_transformers/layers/bert_intermediate.h"
#include "turbo_transformers/layers/bert_output.h"
#include "turbo_transformers/layers/bert_pooler.h"
#include "turbo_transformers/layers/fully_connected.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/utils.h"
#include "turbo_transformers/layers/prepare_bert_masks.h"
#include "turbo_transformers/layers/sequence_pool.h"
#include "turbo_transformers/loaders/npz_load.h"

using namespace turbo_transformers::loaders;

static std::unique_ptr<layers::BERTEmbedding> LoadEmbedding(NPZMapView npz,
                                                            DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BERTEmbedding>(new layers::BERTEmbedding(
      params["word_embeddings.weight"], params["position_embeddings.weight"],
      params["token_type_embeddings.weight"], params["LayerNorm.weight"],
      params["LayerNorm.bias"]));
}

static std::unique_ptr<layers::BertPooler> LoadPooler(NPZMapView npz,
                                                      DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BertPooler>(
      new layers::BertPooler(params["dense.weight"], params["dense.bias"]));
}

static std::unique_ptr<layers::FullyConnected> LoadFnn(
    std::string &&name, NPZMapView npz, layers::types::ActivationType act_type,
    DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);
  return std::unique_ptr<layers::FullyConnected>(new layers::FullyConnected(
      params["weight"], params["bias"], act_type, std::move(name)));
}

struct BERTLayer {
  explicit BERTLayer(NPZLoader params, int64_t n_heads) {
    // define layer network here
    attention_.reset(new layers::BertAttention(
        params["attention.qkv.weight"], params["attention.qkv.bias"],
        params["attention.output.dense.weight"],
        params["attention.output.dense.bias"],
        params["attention.output.LayerNorm.weight"],
        params["attention.output.LayerNorm.bias"], n_heads));
    intermediate_.reset(
        new layers::BertIntermediate(params["intermediate.dense.weight"],
                                     params["intermediate.dense.bias"]));
    output_.reset(new layers::BertOutput(
        params["output.dense.weight"], params["output.dense.bias"],
        params["output.LayerNorm.weight"], params["output.LayerNorm.bias"]));
  }

  void operator()(core::Tensor &hidden, core::Tensor &mask,
                  core::Tensor *attention_out, core::Tensor *intermediate_out,
                  core::Tensor *output) {
    (*attention_)(hidden, mask, attention_out);
    (*intermediate_)(*attention_out, intermediate_out);
    (*output_)(*intermediate_out, *attention_out, output);
  }

  std::unique_ptr<layers::BertAttention> attention_;
  std::unique_ptr<layers::BertIntermediate> intermediate_;
  std::unique_ptr<layers::BertOutput> output_;
};

struct QDRelevanceModel::Impl {
  explicit Impl(const std::string &filename, DLDeviceType device_type,
                size_t n_layers, int64_t n_heads)
      : device_type_(device_type) {
    auto npz = cnpy::npz_load(filename);
    NPZMapView root("", &npz);

    // HERE define your network model
    embedding_ = LoadEmbedding(root.Sub("embeddings"), device_type);

    for (size_t i = 0; i < n_layers; ++i) {
      auto view = root.Sub("encoder.layer." + std::to_string(i));
      NPZLoader params(view, device_type);
      encoders_.emplace_back(std::move(params), n_heads);
    }

    pooler_ = LoadPooler(root.Sub("pooler"), device_type);

    fnn1_ = LoadFnn("fnn1", root.Sub("fnn1"),
                    layers::types::ActivationType::None, device_type);
    fnn2_ = LoadFnn("fnn2", root.Sub("fnn2"),
                    layers::types::ActivationType::Tanh, device_type);
  }

  // preprocess helper function
  template <typename T>
  void PadTensor(const std::vector<std::vector<T>> &data_array, int64_t n,
                 int64_t m, T pad_val, DLDeviceType device_type,
                 core::Tensor *output_tensor) {
    if (m == 0 || n == 0 || data_array.size() == 0) {
      return;
    }
    core::Tensor cpu_tensor(nullptr);
    T *tensor_data_ptr;
    if (device_type == DLDeviceType::kDLGPU) {
      tensor_data_ptr = cpu_tensor.Reshape<T>({n, m}, DLDeviceType::kDLCPU, 0);
      output_tensor->Reshape<T>({n, m}, device_type, 0);
    } else {
      tensor_data_ptr = output_tensor->Reshape<T>({n, m}, device_type, 0);
    }
    for (int64_t i = 0; i < n; ++i, tensor_data_ptr += m) {
      auto &line = data_array[i];
      if (line.size() > 0) {
        core::Copy(line.data(), line.size(), DLDeviceType::kDLCPU,
                   DLDeviceType::kDLCPU, tensor_data_ptr);
      }
      if (line.size() != static_cast<size_t>(m)) {
        layers::kernels::common::Fill(tensor_data_ptr + line.size(),
                                      static_cast<size_t>(m) - line.size(),
                                      pad_val, DLDeviceType::kDLCPU);
      }
    }
    if (device_type == DLDeviceType::kDLGPU) {
      core::Copy<T>(cpu_tensor, *output_tensor);
    }
  }

  // do inference
  std::vector<float> operator()(
      const std::vector<std::vector<int64_t>> &inputs,
      const std::vector<std::vector<int64_t>> &input_masks,
      const std::vector<std::vector<int64_t>> &segment_ids, PoolType pooling) {
    core::Tensor inputs_tensor{nullptr};
    core::Tensor masks_tensor{nullptr};
    int64_t max_seq_len =
        std::accumulate(inputs.begin(), inputs.end(), 0,
                        [](size_t len, const std::vector<int64_t> &input_ids) {
                          return std::max(len, input_ids.size());
                        });
    int64_t batch_size = inputs.size();
    auto *iptr = inputs_tensor.Reshape<int64_t>(
        {batch_size, max_seq_len}, DLDeviceType::kDLCPU, 0,
        "PrepareBertMasks/seqids/Reshape");
    auto *mptr = masks_tensor.Reshape<int64_t>(
        {batch_size, max_seq_len}, DLDeviceType::kDLCPU, 0,
        "PrepareBertMasks/attmask/Reshape");

    for (size_t i = 0; i < inputs.size();
         ++i, iptr += max_seq_len, mptr += max_seq_len) {
      auto &input = inputs[i];
      auto &mask = input_masks[i];
      std::copy(input.begin(), input.end(), iptr);
      std::copy(mask.begin(), mask.end(), mptr);
    }

    auto &inputIds = inputs_tensor;

    core::Tensor seqType(nullptr);
    core::Tensor positionIds(nullptr);
    core::Tensor extendedAttentionMask(nullptr);

    layers::PrepareBertMasks()(inputIds, &masks_tensor, &seqType, &positionIds,
                               &extendedAttentionMask);

    // start inference the BERT
    core::Tensor hidden(nullptr);
    (*embedding_)(inputIds, positionIds, seqType, &hidden);
    core::Tensor attOut(nullptr);
    core::Tensor intermediateOut(nullptr);
    for (auto &layer : encoders_) {
      layer(hidden, extendedAttentionMask, &attOut, &intermediateOut, &hidden);
    }

    core::Tensor cls_layer(nullptr);
    core::Tensor poolingOutput(nullptr);
    layers::SequencePool(static_cast<layers::types::PoolType>(pooling))(
        hidden, &poolingOutput);
    (*pooler_)(poolingOutput, &cls_layer);

    // 加入 nn 网络
    core::Tensor &all_layer = hidden;
    core::Tensor max_layer(nullptr), mean_layer(nullptr);

    MaskAllLayer(input_masks, &all_layer);
    layers::SequencePool(layers::types::PoolType::kMax)(all_layer, &max_layer);
    GetMaskedMean(input_masks, all_layer, &mean_layer);
    core::Tensor fnn1_input(nullptr);

    layers::kernels::Concat<float>(cls_layer, max_layer, mean_layer, 1,
                                   &fnn1_input, "fnn1/nn_input");

    core::Tensor fnn1_output(nullptr);
    (*fnn1_)(fnn1_input, &fnn1_output);

    core::Tensor score_tensor(nullptr);
    (*fnn2_)(fnn1_output, &score_tensor);

    TT_ENFORCE_EQ(score_tensor.shape(0), static_cast<size_t>(batch_size),
                  "score_tensor should have batch_size as dim 0");
    TT_ENFORCE_EQ(score_tensor.shape(1), static_cast<size_t>(1),
                  "score_tensor should have 1 as dim 1");

    std::vector<float> vec;
    vec.resize(score_tensor.numel());
    core::Copy(score_tensor, vec);
    return vec;
  }

  void MaskAllLayer(const std::vector<std::vector<int64_t>> &input_masks,
                    core::Tensor *all_layer) {
    size_t batch_size = all_layer->shape(0);
    size_t seq_len = all_layer->shape(1);
    size_t hidden_size = all_layer->shape(2);
    TT_ENFORCE_EQ(input_masks.size(), static_cast<size_t>(batch_size),
                  "input masks should have the same batch size as all layers");
    TT_ENFORCE_EQ(input_masks[0].size(), static_cast<size_t>(seq_len),
                  "input masks should have the same seq length as all layers");
    float *data_ptr = all_layer->mutableData<float>();
#pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++) {
      auto it = std::lower_bound(input_masks[i].begin(), input_masks[i].end(),
                                 0, std::greater<int64_t>());
      size_t j = it - input_masks[i].begin();
      memset((void *)(data_ptr + (i * seq_len + j) * hidden_size), 0,
             (seq_len - j) * hidden_size);
    }
  }

  inline void ProcessEle(const float *in_ptr, float *out_ptr,
                         int64_t masked_seq_len, int64_t hidden_size) {
    TT_ENFORCE_GT(
        hidden_size, 0,
        "Avg Pooling on tensor whose leading dimension should be larger than "
        "0");
    for (int64_t i = 0; i < hidden_size; ++i) {
      out_ptr[i] = 0.;
      for (int64_t j = i; j < masked_seq_len * hidden_size; j += hidden_size) {
        out_ptr[i] += in_ptr[j];
      }
      out_ptr[i] /= masked_seq_len;
    }
  };

  void GetMaskedMean(const std::vector<std::vector<int64_t>> &input_masks,
                     const core::Tensor &input, core::Tensor *output) {
    size_t batch_size = input.shape(0);
    size_t seq_len = input.shape(1);
    size_t hidden_size = input.shape(2);
    TT_ENFORCE_EQ(input_masks.size(), static_cast<size_t>(batch_size),
                  "input masks should have the same batch size as all layers");
    TT_ENFORCE_EQ(input_masks[0].size(), static_cast<size_t>(seq_len),
                  "input masks should have the same seq length as all layers");
    output->Reshape<float>(
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(hidden_size)},
        input.device_type(), input.device_id(), "MaskedMean");
    float *out_ptr = output->mutableData<float>();
    const float *in_ptr = input.data<float>();
#pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++) {
      auto it = std::lower_bound(input_masks[i].begin(), input_masks[i].end(),
                                 0, std::greater<int64_t>());
      size_t masked_seq_len = it - input_masks[i].begin();
      ProcessEle(in_ptr + i * hidden_size * seq_len, out_ptr + i * hidden_size,
                 masked_seq_len, hidden_size);
    }
  }

  std::unique_ptr<layers::BERTEmbedding> embedding_;
  std::vector<BERTLayer> encoders_;
  std::unique_ptr<layers::BertPooler> pooler_;
  std::unique_ptr<layers::FullyConnected> fnn1_;
  std::unique_ptr<layers::FullyConnected> fnn2_;
  DLDeviceType device_type_;
};

QDRelevanceModel::QDRelevanceModel(const std::string &filename,
                                   DLDeviceType device_type, size_t n_layers,
                                   int64_t n_heads)
    : m_(new Impl(filename, device_type, n_layers, n_heads)) {}

std::vector<float> QDRelevanceModel::operator()(
    const std::vector<std::vector<int64_t>> &inputs,
    const std::vector<std::vector<int64_t>> &input_masks,
    const std::vector<std::vector<int64_t>> &segment_ids,
    PoolType pooling) const {
  return m_->operator()(inputs, input_masks, segment_ids, pooling);
}

QDRelevanceModel::~QDRelevanceModel() = default;
