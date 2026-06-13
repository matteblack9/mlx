// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

/**
 * Quantize KV cache in the layout expected by kivi_scaled_dot_product_attention.
 *
 * KIVI-style layout:
 * - keys are quantized per channel across the sequence dimension. The returned
 *   quantized keys have shape [B, H, D, packed_L].
 * - values are quantized per token across the value dimension. The returned
 *   quantized values have shape [B, H, L, packed_D].
 */
MLX_API std::vector<array> kivi_quantize_kv(
    const array& keys,
    const array& values,
    int key_group_size,
    int value_group_size,
    int bits,
    StreamOrDevice s = {});

/** Computes scaled Q @ dequant(K) from KIVI-style quantized keys. */
MLX_API array kivi_fused_dequantized_matmul(
    const array& queries,
    const array& quantized_keys,
    const array& key_scales,
    const array& key_biases,
    const float scale,
    int key_group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

/** Computes SDPA from KIVI-style quantized K/V caches. */
MLX_API array kivi_scaled_dot_product_attention(
    const array& queries,
    const array& quantized_keys,
    const array& key_scales,
    const array& key_biases,
    const array& quantized_values,
    const array& value_scales,
    const array& value_biases,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    int key_group_size = 64,
    int value_group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
