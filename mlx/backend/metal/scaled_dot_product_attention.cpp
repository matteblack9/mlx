// Copyright © 2024 Apple Inc.
#include <algorithm>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

namespace {

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

void sdpa_full_self_attention_nax(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  using namespace mlx::steel;

  int wm = 4;
  int wn = 1;

  int bd = q.shape(-1);
  int bq = 64;
  int bk = 32;

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool do_causal = do_causal_;
  const bool has_sinks = sinks.has_value();

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  auto& compute_encoder = metal::get_command_encoder(s);

  auto kernel = get_steel_attention_nax_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7);
  }

  // grid는 Q의 shape에서 나온다. (NQ, H, B)가 그대로 커널의 tid.x/y/z에 대응:
  //   x = NQ = ceil(qL / bq) : q.shape(2)(시퀀스 길이)를 BQ로 블록화한 개수
  //   y = H  = q.shape(1)    : 헤드 수
  //   z = B  = q.shape(0)    : 배치
  // 각 threadgroup이 "O(=Q와 같은 모양)의 한 타일"을 담당하므로 일이 Q 기준으로
  // 쪼개진다. K/V 시퀀스 길이(kL)는 grid에 안 들어가고 커널 내부 for 루프
  // (kb = 0..NK)로 순회한다 - grid 차원이 아니라 루프 차원이다.
  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  // threadgroup 내부 구성. Q.shape이 아니라 커널 파라미터 wm/wn로 정해진다.
  // 32 = SIMD group 한 개의 레인 수, wm*wn = threadgroup당 SIMD group 수.
  // (커널 선언부의 max_total_threads_per_threadgroup(WM*WN*32)와 일치)
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// prefill(=긴 시퀀스) attention의 CPU측 디스패처.
// 하는 일: ① 어느 커널 구현(nax/일반)을 쓸지 라우팅 → ② 블록 크기 결정 →
// ③ 커널 이름 조립해 파이프라인 획득 → ④ 파라미터/버퍼 바인딩 →
// ⑤ grid/threadgroup 차원으로 GPU에 dispatch. 실제 attention 수학은 GPU 커널
// (steel_attention.h)이 하고, 여기서는 "무엇을 어떻게 띄울지"만 정한다.
//   q,k,v : 입력 [B, H, L, D]   o : 출력(여기에 결과를 씀)
//   scale : softmax 전 score에 곱할 1/sqrt(D)
//   do_causal_ : causal(인과) 마스크 사용 여부
//   mask/sinks : 선택적 사용자 마스크 / attention sink
void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // [라우팅] M5+ GPU의 Neural Accelerator(nax) 경로를 쓸 수 있으면 그쪽으로 위임.
  // 단 (a) head dim이 80이면 nax 미지원이라 제외, (b) float32는 TF32 허용 시에만
  // (정밀도를 그대로 요구하는 fp32는 nax를 안 씀). half/bf16은 통과.
  if (metal::is_nax_available() && q.shape(3) != 80 &&
      (env::enable_tf32() || q.dtype() != float32)) {
    return sdpa_full_self_attention_nax(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& q = */ q,
        /* const array& k = */ k,
        /* const array& v = */ v,
        /* const float scale = */ scale,
        /* array& o = */ o,
        /* bool do_causal_ = */ do_causal_,
        /* const std::optional<array>& mask = */ mask,
        /* const std::optional<array>& sinks = */ sinks);
  }
  // 여기부터는 일반 steel_attention(simdgroup_matrix 기반) 경로.

  using namespace mlx::steel;

  // [스레드그룹 구성] 한 threadgroup당 SIMD group(warp) 배치. wm·wn=4개 warp,
  // 총 wm·wn·32 = 128 스레드. 커널의 WM/WN 템플릿 인자가 된다.
  int wm = 4;
  int wn = 1;

  // [블록 크기] 커널이 한 번에 처리할 타일 크기 (커널의 BQ/BK/BD 템플릿 인자).
  int bd = q.shape(-1); // head dim. 타일의 dim 폭(BD) = head dim 전체
  int bq = 32; // Q 시퀀스 블록 = 한 번에 32개 쿼리
  int bk = bd < 128 ? 32 : 16; // K 블록. head dim이 작으면 K를 더 크게(32) 잡아
                               // SRAM/연산 균형을 맞춤, 크면(≥128) 16으로 줄임

  // [텐서 모양 추출]  q,k,v 모양 = [B, H, L, D]
  int B = q.shape(0); // batch 크기
  int H = q.shape(1); // (Q) head 개수
  int D = q.shape(3); // head dim (= bd)
  // GQA: Q head는 H개지만 K/V head는 더 적다. 몇 개의 Q head가 1개의 KV head를
  // 공유하는지 = H / (KV head 수). 커널에서 kv_head = q_head / gqa_factor 로 쓰임.
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2); // Q 시퀀스 길이 (쿼리 토큰 수)
  int kL = k.shape(2); // K/V 시퀀스 길이 (키 토큰 수, 캐시 포함)

  // [정렬 플래그] 시퀀스 길이가 블록 크기로 딱 나눠떨어지나? 떨어지면 경계 검사
  // 없는 빠른 경로(load_unsafe)를, 아니면 마지막 부분 블록에서 안전 경로를 쓴다.
  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value(); // 사용자 마스크 존재 여부
  const bool do_causal = do_causal_; // causal 마스크 여부
  const bool has_sinks = sinks.has_value(); // attention sink 존재 여부

  // [function constants] 컴파일된 커널을 "파이프라인 생성 시점"에 추가 특수화하는
  // 값들. 숫자(200/201/300/...)는 커널 쪽 [[function_constant(N)]] 인덱스와 일치
  // (steel_attention.h:11-16). 같은 커널 바이너리에서 align/causal/mask 유무에
  // 따라 죽은 분기를 제거한 전용 버전이 생성된다 → 분기 없는 빠른 코드.
  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301},
      {&has_sinks, MTL::DataType::DataTypeBool, 302}};

  // [커널 이름 조립] steel_attention.metal의 instantiate 매크로가 만든 심볼 이름과
  // 정확히 일치하도록 문자열을 만든다. 예: "steel_attention_float16_bq32_bk16_
  // bd128_wm4_wn1_maskfloat16". 이 이름으로 metallib에서 해당 인스턴스를 찾는다.
  // (mask가 없으면 mask 타입 자리에 q의 타입을 넣어 이름을 채움)
  std::string base_name;
  concatenate(
      base_name,
      "steel_attention_",
      type_to_name(q),
      "_bq",
      bq,
      "_bk",
      bk,
      "_bd",
      bd,
      "_wm",
      wm,
      "_wn",
      wn,
      "_mask",
      type_to_name(has_mask ? *mask : q));

  // [hash 이름] base_name + function constants 조합까지 포함한 고유 키.
  // 같은 base_name이라도 align/causal/mask/sinks 조합이 다르면 다른 파이프라인이
  // 므로, 이 키로 컴파일된 파이프라인을 캐시·조회한다. (t='true', n='false')
  std::string hash_name;
  concatenate(
      hash_name,
      base_name,
      "_align_Q_",
      (align_Q ? 't' : 'n'),
      "_align_K_",
      (align_K ? 't' : 'n'),
      "_has_mask_",
      (has_mask ? 't' : 'n'),
      "_do_causal_",
      (do_causal ? 't' : 'n'),
      "_has_sinks_",
      (has_sinks ? 't' : 'n'));

  // [커맨드 인코더] 이 스트림에 GPU 명령을 적어 넣을 인코더를 얻는다.
  auto& compute_encoder = metal::get_command_encoder(s);

  // [파이프라인 획득] 이름·hash·function constants로 컴파일된 compute pipeline을
  // 가져온다(없으면 컴파일·캐시). JIT 빌드면 여기서 소스를 합쳐 런타임 컴파일,
  // AOT 빌드면 metallib에서 심볼을 찾는다. (jit_kernels.cpp / nojit_kernels.cpp)
  auto kernel = get_steel_attention_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      q,
      bq,
      bk,
      bd,
      wm,
      wn,
      (has_mask ? *mask : q));

  compute_encoder.set_compute_pipeline_state(kernel); // 이 커널을 쓰겠다고 등록

  // [블록 개수] 시퀀스를 블록으로 나눈 총 개수 (올림). NQ = grid.x가 된다.
  const int NQ = (qL + bq - 1) / bq; // Q 블록 수 = ceil(qL / bq)
  const int NK = (kL + bk - 1) / bk; // K 블록 수 = ceil(kL / bk) (커널 루프 횟수)

  // [정렬 블록 경계] 딱 떨어지는 "꽉 찬" 블록의 개수. 이 인덱스의 블록이 곧
  // 부분(나머지) 블록이라, 커널이 여기서 안전 로드로 전환할지 판단한다.
  const int NQ_aligned = qL / bq; // = NQ-1 (안 떨어질 때) 또는 NQ
  const int NK_aligned = kL / bk;

  // [파라미터 구조체] 커널이 buffer(4)로 받는 AttnParams. 모양·블록정보·stride를
  // 한 묶음으로 GPU에 전달한다. 커널은 이 값들로 포인터 오프셋과 경계 검사를 한다.
  AttnParams params{
      /* int B = */ B, // batch
      /* int H = */ H, // head 수
      /* int D = */ D, // head dim

      /* int qL = */ qL, // Q 시퀀스 길이
      /* int kL = */ kL, // K 시퀀스 길이

      /* int gqa_factor = */ gqa_factor, // Q→KV head 매핑 비율
      /* float scale = */ scale, // softmax 전 score scale (1/sqrt(D))

      /* int NQ = */ NQ, // Q 블록 수
      /* int NK = */ NK, // K 블록 수 (커널 kb 루프 한계)

      /* int NQ_aligned = */ NQ_aligned, // 꽉 찬 Q 블록 경계
      /* int NK_aligned = */ NK_aligned, // 꽉 찬 K 블록 경계

      /* int qL_rem = */ (qL - NQ_aligned * bq), // 마지막 Q 블록 실제 행 수(나머지)
      /* int kL_rem = */ (kL - NK_aligned * bk), // 마지막 K 블록 실제 키 수(나머지)
      /* int qL_off = */ (kL - qL), // K가 Q보다 긴 만큼의 정렬 보정(과거 캐시 길이).
                                    // causal에서 Q토큰 절대위치 계산에 쓰임

      // 각 텐서의 [batch, head, seq] stride. 커널이 자기 블록 시작 주소를 계산할 때
      // tid.z·strides[0] + tid.y·strides[1] + ... 로 사용 (steel_attention.h:90~).
      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  // [버퍼 바인딩] 커널 시그니처의 buffer(N)에 실제 데이터를 연결.
  // 인덱스는 steel_attention.h의 [[buffer(N)]] 선언과 정확히 대응한다.
  compute_encoder.set_input_array(q, 0); // buffer(0) Q
  compute_encoder.set_input_array(k, 1); // buffer(1) K
  compute_encoder.set_input_array(v, 2); // buffer(2) V
  compute_encoder.set_output_array(o, 3); // buffer(3) O (출력)
  compute_encoder.set_bytes(params, 4); // buffer(4) AttnParams (값 복사 전달)

  // mask가 있을 때만 buffer(5)/(6) 바인딩 (커널은 function_constant(has_mask)로
  // 이 버퍼들을 조건부 선언하므로, 없으면 바인딩 자체를 생략).
  if (has_mask) {
    auto& m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}}; // mask의 [batch,head,seq] stride

    compute_encoder.set_bytes(mask_params, 5); // buffer(5) mask stride
    compute_encoder.set_input_array(m, 6); // buffer(6) mask 데이터
  }
  // attention sink가 있을 때만 buffer(7) 바인딩.
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 7); // buffer(7) sinks
  }

  // grid는 Q의 shape에서 나온다. (NQ, H, B)가 그대로 커널의 tid.x/y/z에 대응:
  //   x = NQ = ceil(qL / bq) : q.shape(2)(시퀀스 길이)를 BQ로 블록화한 개수
  //   y = H  = q.shape(1)    : 헤드 수
  //   z = B  = q.shape(0)    : 배치
  // 각 threadgroup이 "O(=Q와 같은 모양)의 한 타일"을 담당하므로 일이 Q 기준으로
  // 쪼개진다. K/V 시퀀스 길이(kL)는 grid에 안 들어가고 커널 내부 for 루프
  // (kb = 0..NK)로 순회한다 - grid 차원이 아니라 루프 차원이다.
  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  // threadgroup 내부 구성. Q.shape이 아니라 커널 파라미터 wm/wn로 정해진다.
  // 32 = SIMD group 한 개의 레인 수, wm*wn = threadgroup당 SIMD group 수.
  // (커널 선언부의 max_total_threads_per_threadgroup(WM*WN*32)와 일치)
  MTL::Size group_dims = MTL::Size(32, wm, wn);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks" : "_nosinks";

  // Get the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(gqa_factor, 4);
  compute_encoder.set_bytes(N, 5);
  compute_encoder.set_bytes(k_head_stride, 6);
  compute_encoder.set_bytes(k_seq_stride, 7);
  compute_encoder.set_bytes(v_head_stride, 8);
  compute_encoder.set_bytes(v_seq_stride, 9);

  compute_encoder.set_bytes(scale, 10);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 11 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 13);
    compute_encoder.set_bytes(q_seq_stride, 14);
    compute_encoder.set_bytes(head_stride, 15);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 16);
    compute_encoder.set_bytes(q.shape(1), 17);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector_2pass(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_2pass_1_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int n_simds = gqa_factor * q.shape(2);

  char devc = d.get_architecture().back();
  int N = k.shape(2);
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    if (n_simds >= 4) {
      blocks = 64;
    } else {
      blocks = 32;
    }
  }
  if (int blocks_env = env::get_var("MLX_SDPA_BLOCKS", 0); blocks_env > 0) {
    blocks = blocks_env;
  }
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];
  MTL::Size group_dims(32, gqa_factor, q.shape(2));
  MTL::Size grid_dims(k.shape(1), q.shape(0), blocks);

  // Allocate the intermediates
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, q.dtype(), nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.add_temporary(intermediate);
  compute_encoder.add_temporary(sums);
  compute_encoder.add_temporary(maxs);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  bool has_sinks = sinks.has_value();
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
      {&has_sinks, MTL::DataType::DataTypeBool, 25},
      {&blocks, MTL::DataType::DataTypeInt, 26},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";
  hash_name += has_sinks ? "_sinks_" : "_nosinks_";
  hash_name += std::to_string(blocks);

  // Get the kernel
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  check_kernel_threadgroup_size(kernel, group_dims, hash_name);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(intermediate, 3);
  compute_encoder.set_output_array(sums, 4);
  compute_encoder.set_output_array(maxs, 5);
  compute_encoder.set_bytes(N, 7);
  compute_encoder.set_bytes(k_head_stride, 8);
  compute_encoder.set_bytes(k_seq_stride, 9);
  compute_encoder.set_bytes(v_head_stride, 10);
  compute_encoder.set_bytes(v_seq_stride, 11);
  compute_encoder.set_bytes(scale, 12);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 13 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 15);
    compute_encoder.set_bytes(q_seq_stride, 16);
    compute_encoder.set_bytes(head_stride, 17);
  }
  if (has_sinks) {
    compute_encoder.set_input_array(*sinks, 18);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Final pass
  kname.clear();
  kname = "sdpa_vector_2pass_2_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Get the kernel
  kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);

  // Launch
  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(q.shape(0) * q.shape(1), q.shape(2), 1);
  check_kernel_threadgroup_size(kernel, group_dims, kname);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (is_training) {
    // It's faster for training on Metal to use the unfused SDPA for both
    // forward and backward.
    return true;
  }
  if (output_logsumexp) {
    return true;
  }
  if (s.device == Device::cpu) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);
  const int num_query_heads = q.shape(1);
  const int num_kv_heads = k.shape(1);
  const int gqa_factor = num_query_heads / num_kv_heads;

  const bool sdpa_vector_supported_head_dim =
      query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256);
  const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);

  const bool sdpa_full_supported_mask = !has_mask || has_arr_mask ||
      (query_sequence_length <= key_sequence_length && do_causal);

  const bool supports_sdpa_full = query_sequence_length > 8 &&
      sdpa_full_supported_mask && sdpa_full_supported_head_dim;

  const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim &&
      (query_sequence_length * gqa_factor) <= 32;

  return !(supports_sdpa_full || supports_sdpa_vector);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

void KiviFusedDequantizedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& kq_pre = inputs[1];
  auto& k_scales_pre = inputs[2];
  auto& k_biases_pre = inputs[3];
  auto& out = outputs[0];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  auto copy_unless_row_contiguous = [&copies, &s](const array& arr)
      -> const array& {
    if (arr.flags().row_contiguous) {
      return arr;
    }
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(std::move(arr_copy));
    return copies.back();
  };

  const auto& q = copy_unless_row_contiguous(q_pre);
  const auto& kq = copy_unless_row_contiguous(kq_pre);
  const auto& k_scales = copy_unless_row_contiguous(k_scales_pre);
  const auto& k_biases = copy_unless_row_contiguous(k_biases_pre);

  out.set_data(allocator::malloc(out.nbytes()));

  KiviQKParams params{
      /* int B = */ q.shape(0),
      /* int Hq = */ q.shape(1),
      /* int Hkv = */ kq.shape(1),
      /* int D = */ q.shape(3),
      /* int qL = */ q.shape(2),
      /* int kL = */ out.shape(3),
      /* int key_group_size = */ key_group_size_,
      /* int key_groups = */ k_scales.shape(3),
      /* float scale = */ scale_};

  std::string kernel_name;
  concatenate(
      kernel_name,
      "kivi_fused_dequantized_matmul_",
      type_to_name(out),
      "_gs_",
      key_group_size_,
      "_b_",
      bits_);

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kernel_name);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(kq, 1);
  compute_encoder.set_input_array(k_scales, 2);
  compute_encoder.set_input_array(k_biases, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(params, 5);

  auto nthreads = out.size();
  auto thread_group_size = std::min<NS::UInteger>(
      kernel->maxTotalThreadsPerThreadgroup(), nthreads);
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
  compute_encoder.add_temporaries(std::move(copies));
}

void KiviScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& kq_pre = inputs[1];
  auto& k_scales_pre = inputs[2];
  auto& k_biases_pre = inputs[3];
  auto& vq_pre = inputs[4];
  auto& v_scales_pre = inputs[5];
  auto& v_biases_pre = inputs[6];
  auto& out = outputs[0];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  auto copy_unless_row_contiguous = [&copies, &s](const array& arr)
      -> const array& {
    if (arr.flags().row_contiguous) {
      return arr;
    }
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(std::move(arr_copy));
    return copies.back();
  };

  const auto& q = copy_unless_row_contiguous(q_pre);
  const auto& kq = copy_unless_row_contiguous(kq_pre);
  const auto& k_scales = copy_unless_row_contiguous(k_scales_pre);
  const auto& k_biases = copy_unless_row_contiguous(k_biases_pre);
  const auto& vq = copy_unless_row_contiguous(vq_pre);
  const auto& v_scales = copy_unless_row_contiguous(v_scales_pre);
  const auto& v_biases = copy_unless_row_contiguous(v_biases_pre);

  out.set_data(allocator::malloc(out.nbytes()));

  KiviSDPAParams params{
      /* int B = */ q.shape(0),
      /* int Hq = */ q.shape(1),
      /* int Hkv = */ kq.shape(1),
      /* int D = */ q.shape(3),
      /* int qL = */ q.shape(2),
      /* int kL = */ k_scales.shape(3) * key_group_size_,
      /* int value_dim = */ out.shape(3),
      /* int key_group_size = */ key_group_size_,
      /* int key_groups = */ k_scales.shape(3),
      /* int value_group_size = */ value_group_size_,
      /* int value_groups = */ v_scales.shape(3),
      /* int do_causal = */ do_causal_ ? 1 : 0,
      /* float scale = */ scale_};

  std::string kernel_name;
  concatenate(
      kernel_name,
      (q.shape(2) <= 8 ? "kivi_sdpa_vector_" : "kivi_sdpa_full_"),
      type_to_name(out),
      "_gsk_",
      key_group_size_,
      "_gsv_",
      value_group_size_,
      "_b_",
      bits_);

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kernel_name);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(kq, 1);
  compute_encoder.set_input_array(k_scales, 2);
  compute_encoder.set_input_array(k_biases, 3);
  compute_encoder.set_input_array(vq, 4);
  compute_encoder.set_input_array(v_scales, 5);
  compute_encoder.set_input_array(v_biases, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_bytes(params, 8);

  auto nthreads = q.shape(0) * q.shape(1) * q.shape(2);
  auto thread_group_size = std::min<NS::UInteger>(
      kernel->maxTotalThreadsPerThreadgroup(), nthreads);
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
  compute_encoder.add_temporaries(std::move(copies));
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = outputs[0];

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(inputs.size());
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }
  bool has_arr_mask = inputs.size() > (3 + has_sinks_);

  // We are in vector mode ie single query
  if (q_pre.shape(2) <= 8) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    bool q_copied = !q_copy_unless(q_pre);
    array q = (q_copied) ? contiguous_copy_gpu(q_pre, s) : q_pre;
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      if (q_copied) {
        copies.push_back(q);
      }
      o.set_data(allocator::malloc(o.nbytes()));
    }

    auto mask_copy_unless = [&q](const array& arr) {
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
          (strides[0] == strides[1] * shape[1]);
    };

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(mask_copy_unless, inputs[3])}
        : std::nullopt;

    // We route to the 2 pass fused attention if
    // - The device is large and the sequence length long
    // - The sequence length is even longer and we have gqa
    bool do_causal = do_causal_ && q.shape(2) > 1;
    char devc = d.get_architecture().back();
    if (((devc == 'd' || devc == 's') && k.shape(2) >= 1024) ||
        (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
      sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    } else {
      sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask, sinks);
    }
  }

  // Full attention mode
  else {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(is_matrix_contiguous, inputs[3])}
        : std::nullopt;

    sdpa_full_self_attention_metal(
        s, d, q, k, v, scale_, o, do_causal_, mask, sinks);
  }

  metal::get_command_encoder(s).add_temporaries(std::move(copies));
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return true;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("NYI");
}

} // namespace mlx::core::fast
