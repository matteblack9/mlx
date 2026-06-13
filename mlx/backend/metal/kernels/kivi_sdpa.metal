// Copyright (c) 2026

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

struct KiviQKParams {
  int B;
  int Hq;
  int Hkv;
  int D;
  int qL;
  int kL;
  int key_group_size;
  int key_groups;
  float scale;
};

struct KiviSDPAParams {
  int B;
  int Hq;
  int Hkv;
  int D;
  int qL;
  int kL;
  int value_dim;
  int key_group_size;
  int key_groups;
  int value_group_size;
  int value_groups;
  int do_causal;
  float scale;
};

template <int bits>
METAL_FUNC uint32_t kivi_unpack(
    const device uint8_t* row,
    int value_index,
    int row_bytes) {
  const int bit_offset = value_index * bits;
  const int byte_offset = bit_offset >> 3;
  const int shift = bit_offset & 7;
  uint32_t word = row[byte_offset];
  if (byte_offset + 1 < row_bytes) {
    word |= uint32_t(row[byte_offset + 1]) << 8;
  }
  return (word >> shift) & ((1u << bits) - 1u);
}

template <typename T, int bits>
METAL_FUNC float kivi_dequant_value(
    const device uint8_t* packed,
    const device T* scales,
    const device T* biases,
    int row_index,
    int value_index,
    int group_size,
    int groups) {
  const int expanded = group_size * groups;
  const int row_bytes = (expanded * bits) >> 3;
  const int group = value_index / group_size;
  const uint32_t q = kivi_unpack<bits>(
      packed + row_index * row_bytes, value_index, row_bytes);
  const int meta_index = row_index * groups + group;
  return float(scales[meta_index]) * float(q) + float(biases[meta_index]);
}

template <typename T, int bits>
METAL_FUNC float kivi_qk_score(
    const device T* q,
    const device uint8_t* kq,
    const device T* k_scales,
    const device T* k_biases,
    constant KiviQKParams& params,
    int b,
    int hq,
    int qpos,
    int kpos) {
  const int gqa = params.Hq / params.Hkv;
  const int hkv = hq / gqa;
  float acc = 0.0f;
  for (int d = 0; d < params.D; ++d) {
    const int q_index =
        ((b * params.Hq + hq) * params.qL + qpos) * params.D + d;
    const int k_row = (b * params.Hkv + hkv) * params.D + d;
    const float kval = kivi_dequant_value<T, bits>(
        kq,
        k_scales,
        k_biases,
        k_row,
        kpos,
        params.key_group_size,
        params.key_groups);
    acc += float(q[q_index]) * kval;
  }
  return acc * params.scale;
}

template <typename T, int group_size, int bits>
[[kernel]] void kivi_fused_dequantized_matmul(
    const device T* q [[buffer(0)]],
    const device uint8_t* kq [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant KiviQKParams& params [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const int kpos = index % params.kL;
  const int qpos = (index / params.kL) % params.qL;
  const int hq = (index / (params.kL * params.qL)) % params.Hq;
  const int b = index / (params.kL * params.qL * params.Hq);
  if (b >= params.B) {
    return;
  }

  const float score = kivi_qk_score<T, bits>(
      q, kq, k_scales, k_biases, params, b, hq, qpos, kpos);
  out[index] = T(score);
}

template <typename T, int bits>
METAL_FUNC float kivi_sdpa_score(
    const device T* q,
    const device uint8_t* kq,
    const device T* k_scales,
    const device T* k_biases,
    constant KiviSDPAParams& params,
    int b,
    int hq,
    int qpos,
    int kpos) {
  const int gqa = params.Hq / params.Hkv;
  const int hkv = hq / gqa;
  float acc = 0.0f;
  for (int d = 0; d < params.D; ++d) {
    const int q_index =
        ((b * params.Hq + hq) * params.qL + qpos) * params.D + d;
    const int k_row = (b * params.Hkv + hkv) * params.D + d;
    const float kval = kivi_dequant_value<T, bits>(
        kq,
        k_scales,
        k_biases,
        k_row,
        kpos,
        params.key_group_size,
        params.key_groups);
    acc += float(q[q_index]) * kval;
  }
  return acc * params.scale;
}

template <
    typename T,
    int key_group_size,
    int value_group_size,
    int bits,
    bool vector_mode>
[[kernel]] void kivi_sdpa_reference(
    const device T* q [[buffer(0)]],
    const device uint8_t* kq [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint8_t* vq [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    constant KiviSDPAParams& params [[buffer(8)]],
    uint row [[thread_position_in_grid]]) {
  const int qpos = row % params.qL;
  const int hq = (row / params.qL) % params.Hq;
  const int b = row / (params.qL * params.Hq);
  if (b >= params.B) {
    return;
  }

  constexpr int max_value_dim = 256;
  float acc[max_value_dim];
  for (int d = 0; d < params.value_dim; ++d) {
    acc[d] = 0.0f;
  }

  const int q_abs = params.kL - params.qL + qpos;
  float max_score = -INFINITY;
  for (int kpos = 0; kpos < params.kL; ++kpos) {
    if (params.do_causal && kpos > q_abs) {
      continue;
    }
    const float score = kivi_sdpa_score<T, bits>(
        q, kq, k_scales, k_biases, params, b, hq, qpos, kpos);
    max_score = metal::max(max_score, score);
  }

  float denom = 0.0f;
  for (int kpos = 0; kpos < params.kL; ++kpos) {
    if (params.do_causal && kpos > q_abs) {
      continue;
    }
    const float score = kivi_sdpa_score<T, bits>(
        q, kq, k_scales, k_biases, params, b, hq, qpos, kpos);
    denom += metal::exp(score - max_score);
  }

  const int gqa = params.Hq / params.Hkv;
  const int hkv = hq / gqa;
  for (int kpos = 0; kpos < params.kL; ++kpos) {
    if (params.do_causal && kpos > q_abs) {
      continue;
    }
    const float score = kivi_sdpa_score<T, bits>(
        q, kq, k_scales, k_biases, params, b, hq, qpos, kpos);
    const float weight = metal::exp(score - max_score) / denom;
    for (int d = 0; d < params.value_dim; ++d) {
      const int v_row = (b * params.Hkv + hkv) * params.kL + kpos;
      const float vval = kivi_dequant_value<T, bits>(
          vq,
          v_scales,
          v_biases,
          v_row,
          d,
          params.value_group_size,
          params.value_groups);
      acc[d] += weight * vval;
    }
  }

  const int out_base =
      ((b * params.Hq + hq) * params.qL + qpos) * params.value_dim;
  for (int d = 0; d < params.value_dim; ++d) {
    out[out_base + d] = T(acc[d]);
  }
}

#define instantiate_kivi_qk(type, type_name, group_size, bits)       \
  instantiate_kernel(                                                \
      "kivi_fused_dequantized_matmul_" #type_name "_gs_" #group_size \
      "_b_" #bits,                                                   \
      kivi_fused_dequantized_matmul,                                 \
      type,                                                          \
      group_size,                                                    \
      bits)

#define instantiate_kivi_sdpa(                                                   \
    name, type, type_name, key_group_size, value_group_size, bits, vector_mode)  \
  instantiate_kernel(                                                            \
      #name "_" #type_name "_gsk_" #key_group_size "_gsv_" #value_group_size     \
      "_b_" #bits,                                                               \
      kivi_sdpa_reference,                                                       \
      type,                                                                      \
      key_group_size,                                                            \
      value_group_size,                                                          \
      bits,                                                                      \
      vector_mode)

#define instantiate_kivi_all(type, type_name, bits)                         \
  instantiate_kivi_qk(type, type_name, 32, bits)                            \
  instantiate_kivi_qk(type, type_name, 64, bits)                            \
  instantiate_kivi_qk(type, type_name, 128, bits)                           \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 32, 32, bits, true)   \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 32, 64, bits, true)   \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 32, 128, bits, true)  \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 64, 32, bits, true)   \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 64, 64, bits, true)   \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 64, 128, bits, true)  \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 128, 32, bits, true)  \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 128, 64, bits, true)  \
  instantiate_kivi_sdpa(kivi_sdpa_vector, type, type_name, 128, 128, bits, true) \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 32, 32, bits, false)    \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 32, 64, bits, false)    \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 32, 128, bits, false)   \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 64, 32, bits, false)    \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 64, 64, bits, false)    \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 64, 128, bits, false)   \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 128, 32, bits, false)   \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 128, 64, bits, false)   \
  instantiate_kivi_sdpa(kivi_sdpa_full, type, type_name, 128, 128, bits, false)

#define instantiate_kivi_bits(type, type_name) \
  instantiate_kivi_all(type, type_name, 2)     \
  instantiate_kivi_all(type, type_name, 3)     \
  instantiate_kivi_all(type, type_name, 4)     \
  instantiate_kivi_all(type, type_name, 5)     \
  instantiate_kivi_all(type, type_name, 6)     \
  instantiate_kivi_all(type, type_name, 8)

instantiate_kivi_bits(float, float)
instantiate_kivi_bits(float16_t, float16)
instantiate_kivi_bits(bfloat16_t, bfloat16)
