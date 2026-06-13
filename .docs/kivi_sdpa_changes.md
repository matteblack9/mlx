# KIVI SDPA 변경 기록

이 문서는 KIVI 방식의 KV quantization 및 `Q @ dequant(K)` 경로를 MLX 코드베이스에 붙이기 위해 추가한 변경 사항을 정리한다.

현재 구현의 목적은 다음과 같다.

- 기존 SDPA 구현을 크게 건드리지 않고 KIVI-style quantized KV cache 경로를 노출한다.
- 사용자가 실제 Metal fused kernel을 작성하기 전에, Python/C++ API와 benchmark 진입점을 먼저 고정한다.
- decode(`q_len == 1`)와 prefill/full attention(`q_len > 1`) 모두 synthetic benchmark에서 실행되게 한다.

중요한 제한 사항:

- forward 실행에서 `kivi_scaled_dot_product_attention`은 더 이상 기존 `scaled_dot_product_attention` fallback을 호출하지 않는다.
- 현재 Metal kernel은 성능용 kernel이 아니라 reference/skeleton kernel이다. quantized K/V에서 직접 dequantize하며 계산하지만 tile 최적화는 되어 있지 않다.
- array mask와 attention sinks는 아직 KIVI Metal skeleton에 연결하지 않았다. `causal`과 no-mask 경로만 지원한다.
- `value_dim > 256`은 현재 skeleton에서 막아두었다. placeholder kernel이 thread-local accumulator를 `float[256]`으로 둔다.
- 네가 최종적으로 주로 수정할 위치는 `mlx/backend/metal/kernels/kivi_sdpa.metal`이다.

현재 forward 경로:

```text
Python mx.fast.kivi_scaled_dot_product_attention
  -> C++ fast::kivi_scaled_dot_product_attention
  -> KiviScaledDotProductAttention primitive
  -> KiviScaledDotProductAttention::eval_gpu
  -> kivi_sdpa_vector_* 또는 kivi_sdpa_full_* Metal kernel
```

`Q @ dequant(K)` 단독 benchmark 경로:

```text
Python mx.fast.kivi_fused_dequantized_matmul
  -> C++ fast::kivi_fused_dequantized_matmul
  -> KiviFusedDequantizedMatmul primitive
  -> KiviFusedDequantizedMatmul::eval_gpu
  -> kivi_fused_dequantized_matmul_* Metal kernel
```

## 변경 파일 요약

### `mlx/fast.h`

위치: `mlx/fast.h`

추가한 public C++ API는 세 개다.

```cpp
std::vector<array> kivi_quantize_kv(...);
array kivi_fused_dequantized_matmul(...);
array kivi_scaled_dot_product_attention(...);
```

각 함수의 역할은 다음과 같다.

- `kivi_quantize_kv`
  - 입력 K/V shape: `[B, H, L, D]`
  - K는 KIVI 방식으로 per-channel quantization을 하기 위해 `[B, H, D, L]`로 축을 바꾼 뒤 quantize한다.
  - V는 기존 `[B, H, L, D]` layout에서 마지막 축 `D`를 기준으로 quantize한다.
  - 반환값은 `[kq, k_scales, k_biases, vq, v_scales, v_biases]` 순서다.

- `kivi_fused_dequantized_matmul`
  - KIVI quantized K에 대해 scaled `Q @ dequant(K)`를 계산하는 entry point다.
  - 현재는 `KiviFusedDequantizedMatmul` primitive를 만들고, `kivi_fused_dequantized_matmul_*` Metal kernel을 dispatch한다.
  - KIVI K layout이 `[B, Hkv, D, packed_L]`이기 때문에 kernel 내부에서 expanded `[D, L]`로 dequantize하며 query `[B, Hq, qL, D]`와 곱해 `[B, Hq, qL, L]` scores를 만든다.
  - GQA/MQA처럼 `q_heads != kv_heads`인 경우 kernel 내부에서 `kv_head = q_head / (Hq / Hkv)`로 매핑한다. 더 이상 C++ wrapper에서 K/scales/biases를 repeat하지 않는다.

- `kivi_scaled_dot_product_attention`
  - KIVI quantized K/V를 받아 SDPA 결과를 반환하는 wrapper다.
  - 현재는 `KiviScaledDotProductAttention` primitive를 만들고, `qL <= 8`이면 vector kernel, 그 외에는 full kernel을 dispatch한다.
  - forward path에서 dense K/V dequantized tensor를 만들지 않는다.
  - array mask와 sinks 인자는 API에는 남아 있지만 skeleton에서는 아직 unsupported로 명시적으로 예외를 던진다.

### `mlx/fast.cpp`

위치: `mlx/fast.cpp`

기존 `scaled_dot_product_attention` 함수 바로 뒤에 KIVI 관련 구현을 추가했다.

#### `kivi_quantize_kv`

검증:

- `keys`와 `values`가 rank 4인지 확인한다.
- `[B, H, L]` 차원이 서로 같은지 확인한다.
- `key_group_size`, `value_group_size`가 양수인지 확인한다.

처리 흐름:

```cpp
auto keys_by_channel = swapaxes(keys, -1, -2, s);
auto qk = quantize(keys_by_channel, key_group_size, bits, "affine", {}, s);
auto qv = quantize(values, value_group_size, bits, "affine", {}, s);
```

결과 layout:

- `kq`: `[B, H, D, packed_L]`
- `k_scales`, `k_biases`: K quantization group metadata
- `vq`: `[B, H, L, packed_D]`
- `v_scales`, `v_biases`: V quantization group metadata

#### `kivi_fused_dequantized_matmul`

검증:

- `queries`와 `quantized_keys`가 rank 4인지 확인한다.
- `queries.shape(0) == quantized_keys.shape(0)`인지 확인한다.
- `queries.shape(3) == quantized_keys.shape(2)`인지 확인한다.
  - query: `[B, Hq, Lq, D]`
  - quantized key: `[B, Hkv, D, packed_L]`
- `key_group_size`가 양수인지 확인한다.
- `Hq`가 `Hkv`의 배수인지 확인한다.

GQA 처리:

- C++ front door는 `Hq % Hkv == 0`만 검증한다.
- K/scales/biases를 C++에서 repeat하지 않는다.
- Metal kernel 내부에서 `kv_head = q_head / (Hq / Hkv)`로 매핑한다.

현재 계산:

```cpp
auto primitive = std::make_shared<KiviFusedDequantizedMatmul>(
    stream, fallback, scale, key_group_size, bits);
return array(out_shape, final_type, primitive, std::move(inputs));
```

forward 실행은 `KiviFusedDequantizedMatmul::eval_gpu`로 가고, 여기서 `kivi_fused_dequantized_matmul_*` Metal kernel을 dispatch한다. `fallback`은 `Custom` transform용으로만 보관한다.

#### `kivi_scaled_dot_product_attention`

검증:

- `queries`가 rank 4인지 확인한다.
- key/value group size가 양수인지 확인한다.
- array mask와 sinks는 아직 unsupported로 막는다.
- quantized K/V dtype이 `uint32`인지 확인한다.
- q/k/v shape metadata가 일관적인지 확인한다.
- `value_dim <= 256`인지 확인한다.

현재 forward 흐름:

```cpp
auto primitive = std::make_shared<KiviScaledDotProductAttention>(
    stream,
    fallback,
    scale,
    do_causal,
    key_group_size,
    value_group_size,
    bits);
return array(out_shape, final_type, primitive, std::move(inputs));
```

forward 실행은 `KiviScaledDotProductAttention::eval_gpu`로 가고, 여기서 `q.shape(2) <= 8`이면 `kivi_sdpa_vector_*`, 아니면 `kivi_sdpa_full_*` Metal kernel을 dispatch한다.

fallback 함수는 아직 남아 있지만 일반 forward에서 호출되지 않는다. `Custom` transform(vmap/jvp/vjp)용 reference로 남겨둔 것이다.

현재 Metal kernel은 correctness/skeleton용이다. 실제 성능 kernel을 넣을 때는 C++ API를 더 건드리지 않고 `mlx/backend/metal/kernels/kivi_sdpa.metal`의 kernel 본문을 바꾸는 것이 목표다.

### `mlx/fast_primitives.h`

위치: `mlx/fast_primitives.h`

KIVI용 `Custom` primitive 두 개를 추가했다.

```cpp
class KiviFusedDequantizedMatmul : public Custom { ... };
class KiviScaledDotProductAttention : public Custom { ... };
```

#### `KiviFusedDequantizedMatmul`

상태:

- `scale_`
- `key_group_size_`
- `bits_`

역할:

- `kivi_fused_dequantized_matmul` API의 forward 실행을 담당한다.
- `eval_gpu`는 `mlx/backend/metal/scaled_dot_product_attention.cpp`에 구현했다.
- output shape는 `[B, Hq, qL, kL]`이다.

#### `KiviScaledDotProductAttention`

상태:

- `scale_`
- `do_causal_`
- `key_group_size_`
- `value_group_size_`
- `bits_`

역할:

- `kivi_scaled_dot_product_attention` API의 forward 실행을 담당한다.
- `eval_gpu`는 `mlx/backend/metal/scaled_dot_product_attention.cpp`에 구현했다.
- output shape는 `[B, Hq, qL, value_dim]`이다.

주의:

- 두 primitive 모두 `Custom` fallback을 보관한다.
- 하지만 일반 forward에서는 `eval_gpu`가 실행되므로 fallback을 타지 않는다.
- fallback은 transform path(vmap/jvp/vjp)용 reference로 남겨둔 것이다.

### `mlx/backend/metal/scaled_dot_product_attention.cpp`

위치: `mlx/backend/metal/scaled_dot_product_attention.cpp`

KIVI primitive의 Metal dispatch 구현을 추가했다.

추가한 param struct:

```cpp
struct KiviQKParams { ... };
struct KiviSDPAParams { ... };
```

추가한 `eval_gpu`:

```cpp
void KiviFusedDequantizedMatmul::eval_gpu(...);
void KiviScaledDotProductAttention::eval_gpu(...);
```

`KiviFusedDequantizedMatmul::eval_gpu`:

- q/kq/scales/biases를 row-contiguous로 보장한다.
- output buffer를 allocate한다.
- kernel name:

```text
kivi_fused_dequantized_matmul_{dtype}_gs_{key_group_size}_b_{bits}
```

- dispatch thread 수:

```text
B * Hq * qL * kL
```

`KiviScaledDotProductAttention::eval_gpu`:

- q/kq/ks/kb/vq/vs/vb를 row-contiguous로 보장한다.
- output buffer를 allocate한다.
- `qL <= 8`이면 vector kernel name을 사용한다.
- `qL > 8`이면 full kernel name을 사용한다.

```text
kivi_sdpa_vector_{dtype}_gsk_{key_group_size}_gsv_{value_group_size}_b_{bits}
kivi_sdpa_full_{dtype}_gsk_{key_group_size}_gsv_{value_group_size}_b_{bits}
```

- dispatch thread 수:

```text
B * Hq * qL
```

즉 한 thread가 한 query row `(B, Hq, qpos)`를 담당하고, 현재 placeholder kernel은 그 thread 안에서 `value_dim` 전체를 순차로 쓴다.

### `mlx/backend/metal/kernels/kivi_sdpa.metal`

위치: `mlx/backend/metal/kernels/kivi_sdpa.metal`

새로 추가한 Metal kernel skeleton 파일이다. 네가 최종적으로 가장 많이 수정할 파일이다.

포함된 kernel:

```metal
kivi_fused_dequantized_matmul(...)
kivi_sdpa_reference(...)
```

host name으로 노출되는 kernel:

```text
kivi_fused_dequantized_matmul_{dtype}_gs_{group_size}_b_{bits}
kivi_sdpa_vector_{dtype}_gsk_{key_group_size}_gsv_{value_group_size}_b_{bits}
kivi_sdpa_full_{dtype}_gsk_{key_group_size}_gsv_{value_group_size}_b_{bits}
```

지원 dtype:

- `float`
- `float16`
- `bfloat16`

지원 bits:

- `2`
- `3`
- `4`
- `5`
- `6`
- `8`

지원 group size:

- key group size: `32`, `64`, `128`
- value group size: `32`, `64`, `128`

현재 placeholder kernel 구조:

- `kivi_unpack<bits>`:
  - MLX affine quantization의 packed bitstream에서 quantized integer를 읽는다.
  - `uint32` storage를 Metal에서는 `uint8_t*`로 읽는다.

- `kivi_dequant_value<T, bits>`:
  - packed value, scale, bias를 사용해 scalar 하나를 dequantize한다.

- `kivi_fused_dequantized_matmul`:
  - 1 thread가 score 하나 `[B,Hq,qL,kL]`를 계산한다.
  - 내부에서 K를 scalar 단위로 dequantize하며 dot product를 계산한다.

- `kivi_sdpa_reference`:
  - 1 thread가 output row 하나 `[B,Hq,qL,:]`를 계산한다.
  - score max pass, denominator pass, output accumulation pass를 수행한다.
  - K/V dense tensor를 만들지 않는다.
  - 성능 최적화가 전혀 되어 있지 않은 reference skeleton이다.

실제 커널 개발 시 바꿀 곳:

- QK만 최적화하려면 `kivi_fused_dequantized_matmul` 본문을 교체한다.
- end-to-end SDPA를 최적화하려면 `kivi_sdpa_reference`를 대체하거나, vector/full용 별도 고성능 kernel을 만들고 host name만 유지한다.
- C++/Python/benchmark 쪽 API는 그대로 유지하면 된다.

### `mlx/backend/metal/kernels/CMakeLists.txt`

위치: `mlx/backend/metal/kernels/CMakeLists.txt`

추가:

```cmake
build_kernel(kivi_sdpa)
```

이 등록이 있어야 `kivi_sdpa.metal`이 `mlx.metallib`에 포함되고, C++에서 `d.get_kernel(kernel_name)`으로 로드할 수 있다.

### `python/src/fast.cpp`

위치: `python/src/fast.cpp`

Python `mx.fast` namespace에 세 함수를 바인딩했다.

#### `mx.fast.kivi_quantize_kv`

Python signature:

```python
mx.fast.kivi_quantize_kv(
    k,
    v,
    *,
    key_group_size=64,
    value_group_size=64,
    bits=4,
    stream=None,
)
```

반환값:

```python
[kq, k_scales, k_biases, vq, v_scales, v_biases]
```

#### `mx.fast.kivi_fused_dequantized_matmul`

Python signature:

```python
mx.fast.kivi_fused_dequantized_matmul(
    q,
    kq,
    k_scales,
    k_biases,
    *,
    scale,
    key_group_size=64,
    bits=4,
    stream=None,
)
```

용도:

- KIVI quantized K에 대한 `Q @ dequant(K)`만 따로 benchmark하기 위한 함수다.
- 실제 fused Metal kernel을 작성할 때 가장 먼저 교체/비교할 대상이다.

#### `mx.fast.kivi_scaled_dot_product_attention`

Python signature:

```python
mx.fast.kivi_scaled_dot_product_attention(
    q,
    kq,
    k_scales,
    k_biases,
    vq,
    v_scales,
    v_biases,
    *,
    scale,
    mask=None,
    sinks=None,
    key_group_size=64,
    value_group_size=64,
    bits=4,
    stream=None,
)
```

mask 처리:

- `None`: no mask
- `"causal"`: KIVI Metal skeleton의 causal path
- `array`: Python binding에서는 받을 수 있지만 현재 KIVI Metal skeleton에서는 C++에서 unsupported 예외를 던진다.

문자열 mask는 `"causal"`만 허용한다. 다른 문자열은 `invalid mask option` 예외를 던진다.

### `benchmarks/python/kivi_sdpa_bench.py`

위치: `benchmarks/python/kivi_sdpa_bench.py`

기존 benchmark 디렉토리 구조에 맞춰 KIVI SDPA 전용 Python benchmark를 추가했다.

실행 예:

```bash
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --mask causal --iters 10
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --mask none --iters 10
```

옵션:

```bash
--bits
--key-group-size
--value-group-size
--iters
--mask {none,causal}
--case {all,prefill,decode}
```

기본 case:

- prefill/full:
  - `B=1`
  - `qL=128`
  - `kL=128`
  - `D=64`
  - `qH=8`
  - `kH=4`

- decode:
  - `B=1`
  - `qL=1`
  - `kL=256`
  - `D=64`
  - `qH=8`
  - `kH=4`

benchmark가 비교하는 항목:

- `qk_ref`
  - `dequantize(kq) -> matmul(q, k)` reference
- `qk_fused`
  - `mx.fast.kivi_fused_dequantized_matmul`
  - 현재 forward는 `KiviFusedDequantizedMatmul` primitive와 `kivi_fused_dequantized_matmul_*` Metal kernel을 탄다.
- `attn_ref`
  - `dequantize(kq/vq) -> mx.fast.scaled_dot_product_attention`
- `attn_kivi`
  - `mx.fast.kivi_scaled_dot_product_attention`
  - 현재 forward는 `KiviScaledDotProductAttention` primitive와 `kivi_sdpa_vector_*` 또는 `kivi_sdpa_full_*` Metal kernel을 탄다.

출력 예:

```text
prefill  ... qk_abs=3.906e-03 attn_abs=9.766e-04 qk_ref=...ms qk_fused=...ms attn_ref=...ms attn_kivi=...ms
decode   ... qk_abs=1.953e-03 attn_abs=2.441e-04 qk_ref=...ms qk_fused=...ms attn_ref=...ms attn_kivi=...ms
```

해석:

- `qk_abs`
  - dequantized reference와 KIVI QK Metal skeleton의 최대 절대 오차다.
  - 현재 placeholder kernel은 scalar float accumulation 후 output dtype으로 cast한다. reference matmul과 accumulation 경로가 달라 `1e-3` 수준 오차가 날 수 있다.

- `attn_abs`
  - dequantized SDPA reference와 KIVI SDPA Metal skeleton의 최대 절대 오차다.
  - 현재 placeholder kernel은 correctness/skeleton용이므로 기존 SDPA와 bitwise identical하지 않다.

### root `benchmark.py`

초기에는 root에 임시 `benchmark.py`를 만들었지만, 이 repo에는 이미 `benchmarks/` 디렉토리 구조가 있으므로 제거했다.

최종 benchmark entry는 다음 파일만 사용한다.

```text
benchmarks/python/kivi_sdpa_bench.py
```

### `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`

기존 파일에 있던 syntax-breaking stray backtick 하나를 제거했다.

변경 의도:

- 내가 추가한 KIVI API 빌드를 확인하는 과정에서 Metal/Steel attention header가 같이 컴파일됐다.
- 해당 파일에 stray backtick이 있어 빌드 실패 가능성이 있었기 때문에 문법만 고쳤다.
- SDPA algorithm이나 tile scheduling 로직은 바꾸지 않았다.

주의:

- 이 파일에는 사용자가 이전에 추가한 설명 주석이 많이 들어 있다.
- 그 주석들은 되돌리지 않았다.

## 빌드 및 검증 기록

빌드:

```bash
python3.11 setup.py build_ext --inplace
```

검증:

```bash
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --mask causal --iters 1
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --mask none --iters 1
```

검증 결과:

- causal mask:
  - prefill 실행 성공
  - decode 실행 성공
  - prefill: `qk_abs=3.906e-03`, `attn_abs=9.766e-04`
  - decode: `qk_abs=1.953e-03`, `attn_abs=2.441e-04`
- no mask:
  - prefill 실행 성공
  - decode 실행 성공
  - prefill: `qk_abs=3.906e-03`, `attn_abs=4.883e-04`
  - decode: `qk_abs=1.953e-03`, `attn_abs=2.441e-04`

성능 주의:

- 현재 `attn_kivi`는 reference/skeleton Metal kernel이라 기존 SDPA보다 느리다.
- 이 수치는 correctness 및 dispatch 연결 확인용이고, 성능 판단용이 아니다.

## 이제 커널만 개발하면 되는 지점

현재 C++/Python/benchmark/Metal dispatch skeleton은 연결되어 있다. 일반 forward는 기존 SDPA fallback을 타지 않는다.

네가 주로 수정할 파일:

```text
mlx/backend/metal/kernels/kivi_sdpa.metal
```

QK만 최적화할 때:

- 수정 대상: `kivi_fused_dequantized_matmul`
- host name은 유지한다.

```text
kivi_fused_dequantized_matmul_{dtype}_gs_{group_size}_b_{bits}
```

end-to-end SDPA를 최적화할 때:

- 수정 대상: `kivi_sdpa_reference`
- 또는 vector/full 별도 kernel을 만들되 아래 host name을 유지한다.

```text
kivi_sdpa_vector_{dtype}_gsk_{key_group_size}_gsv_{value_group_size}_b_{bits}
kivi_sdpa_full_{dtype}_gsk_{key_group_size}_gsv_{value_group_size}_b_{bits}
```

kernel 입력 계약:

- `q`: `[B, Hq, qL, D]`, floating
- `kq`: `[B, Hkv, D, packed_L]`, `uint32` storage, Metal에서는 `uint8_t*`로 읽음
- `k_scales`: `[B, Hkv, D, kL / key_group_size]`
- `k_biases`: `[B, Hkv, D, kL / key_group_size]`
- `vq`: `[B, Hkv, kL, packed_D]`, `uint32` storage, Metal에서는 `uint8_t*`로 읽음
- `v_scales`: `[B, Hkv, kL, value_dim / value_group_size]`
- `v_biases`: `[B, Hkv, kL, value_dim / value_group_size]`
- `out`: `[B, Hq, qL, value_dim]`

GQA/MQA mapping:

```text
kv_head = q_head / (Hq / Hkv)
```

현재 skeleton의 제한:

- array mask 미지원
- attention sinks 미지원
- `value_dim <= 256`
- 성능 최적화 없음

fallback 확인:

- `kivi_scaled_dot_product_attention` forward는 `KiviScaledDotProductAttention::eval_gpu`로 들어간다.
- `kivi_fused_dequantized_matmul` forward는 `KiviFusedDequantizedMatmul::eval_gpu`로 들어간다.
- `Custom` fallback 함수는 transform용 reference로 남아 있지만 일반 benchmark forward에서는 호출되지 않는다.

검증 명령:

```bash
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --case decode --mask causal --iters 1
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --case prefill --mask causal --iters 1
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --case decode --mask none --iters 1
PYTHONPATH=python python3.11 benchmarks/python/kivi_sdpa_bench.py --case prefill --mask none --iters 1
```

## 현재 남아 있는 비관련 변경

`git status`에는 다음 파일들도 modified로 보인다.

```text
mlx/backend/metal/scaled_dot_product_attention.cpp
mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal
mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h
```

이 중 `steel_attention.h`의 stray backtick 제거를 제외한 대량 주석 변경은 사용자가 이미 작업해둔 상태로 보이며, 되돌리지 않았다.
