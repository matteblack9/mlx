// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
// 이 코드는 SIMT 실행 모델이다.
// 커널 함수 전체(이 코드)가 애초에 lane(스레드) 마다 한번씩 통째로 실행된다고 생각하면 된다.
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// ============================================================================
// 용어 정리 & 하드웨어 고정값 vs 튜닝 파라미터
// ============================================================================
// [용어]
//   - SIMD group (= warp) : 함께 lock-step으로 도는 스레드 묶음. 한 단위가 32 lane.
//   - lane                : SIMD group 안의 스레드 하나. simd_lane_id = 0..31.
//   - MMA (Matrix Multiply-Accumulate) : 행렬곱-누적 D=A·B+C 를 한 명령에 하는
//                           HW 기능. Metal에서는 simdgroup_matrix로 노출.
//   - fragment (frag)     : MMA 한 명령이 다루는 최소 행렬 블록. 여기선 8×8.
//                           8×8=64원소를 32 lane이 나눠(lane당 2개) register에 보관.
//   - tile (MMATile)      : fragment들을 격자로 이어붙인 큰 행렬 (register 거주).
//   - threadgroup memory  : on-chip SRAM. threadgroup 내 전 스레드가 공유.
//   - bank                : 그 SRAM을 동시 접근 가능하게 나눈 병렬 "창구".
//                           같은 bank에 여러 lane이 몰리면 직렬화(bank conflict).
//   - 16B 벡터 로드/스토어 : 로더·MMA가 한 트랜잭션에 옮기는 정렬 단위 = 128bit.
//                           T가 half면 8개, float면 4개 원소 (= 16 / sizeof(T)).
//
// [하드웨어가 고정해 둔 값 — 코드에서 바꿀 수 없음]
//   - SIMD group 크기            = 32 lane                 (아래 ...*32, simd_lane_id)
//   - simdgroup_matrix fragment  = 8×8                     (kFragSize=8, MMAFrag)
//   - 정렬 벡터 접근 폭           = 16 byte (128 bit)        (pad = 16/sizeof(T))
//   - SRAM bank 구성(폭·개수)     : bank이 여러 개로 나뉘어 병렬 접근된다는 "성질"은
//                                  HW 고정. 단 정확한 폭/개수는 아키텍처 의존이고
//                                  공개 스펙이 제한적(문서/그림에선 4B×32 가정).
//
// [반대로, 아래는 HW 고정이 아니라 호스트가 고르는 "튜닝 파라미터" (템플릿 인자)]
//   - BQ/BK/BD : Q블록·K블록·head dim 타일 크기   (scaled_dot_product_attention.cpp)
//   - WM/WN    : threadgroup당 SIMD group을 M/N 방향으로 몇 개 (여기선 4×1)
//   - T/MaskType/AccumType : 데이터 타입 / 마스크 타입 / 누적 타입
//   * 이 값들이 컴파일 타임 상수라서 fragment·tile 격자 크기, SRAM 배열 크기, 루프
//     횟수가 전부 컴파일 시점에 확정된다(아래 4줄 참고 = 레지스터 거주/언롤의 근거).
// ============================================================================

// clang-format off
// 템플릿 변수로 값을 둠으로써, 상수기 때문에 명령어 자체가 되고, 레지스터 계산 시, 값을 메모리로 가져올 필요가 없다
// -> 왜냐하면 몇번 레지스터를 읽을지 기계어 명령에 미리 새겨있지 않으면, 레지스터는 변수로 골라서 접근하는 것이 하드웨어적으로 불가능하게 띠문에
// 런타임 인덱스 접근이 하나라도 있으면 그 배열 전체를 주소로 접근 가능한 DRAM에 spill한다.
// STEEL_PRAGMA_UNROLL 같은 것이 가능하게 되며, 정적메모리 크기를 결정할 수 있게된다.
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]], // 0..31 (SIMD group=32 고정)
    uint simd_group_id [[simdgroup_index_in_threadgroup]], // 0..WM*WN-1 (warp 번호)
    uint3 tid [[threadgroup_position_in_grid]], // 블록 인덱스 (grid=(NQ,H,B))
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on
  // max_total_threads_per_threadgroup(WM*WN*32): 여기 '32'가 HW 고정값(SIMD group
  // 크기). threadgroup당 총 스레드 = warp수(WM*WN) × 32.

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  // tid는 thread id가 아니라 threadgroup_position_in_grid(=블록 인덱스, CUDA의
  // blockIdx에 해당)다. grid가 (Q시퀀스블록 수, 헤드 수, 배치 수) 3차원으로
  // 나뉘어 있고, tid.x/y/z로 "이 threadgroup이 어느 배치/헤드/Q블록을
  // 담당하는가"를 정해 각 텐서 포인터를 그 데이터 영역으로 이동시킨다.
  //
  // tidl은 tid를 ulong3로 올린 것 - stride 곱셈에서 32비트 오버플로를 막으려고
  // 64비트로 캐스팅한 것이다.
  // group_dims(32, wm, wn) 은 simd group을 어떻게 나눌 것인가임 -> warp 수 = wm * wn
  ulong3 tidl{tid.x, tid.y, tid.z};


  /**
   *  Q_strides[0] -> batch 하나 넘어갈 때 건너뛸 원소 수
      Q_strides[1] -> head 하나 넘어갈 때 건너뛸 원소 수
      Q_strides[2] -> 토큰 하나 넘어갈 때 건너뛸 원소 수
      (D)	D (head dim)	=1 (저장 안 함, contiguous)
      Q_strides[2]는 **"토큰 개수"가 아니라 "한 토큰에서 다음 토큰으로 갈 때의 보폭"**
  *
  */

  Q += tidl.z * params->Q_strides[0] + // Batch  (tid.z = 배치 인덱스)
      tidl.y * params->Q_strides[1] + // Head   (tid.y = 헤드 인덱스)
      tidl.x * BQ * params->Q_strides[2]; // Sequence (tid.x = Q 시퀀스 블록 인덱스)

  // GQA: 여러 Q 헤드가 하나의 KV 헤드를 공유한다. Q 헤드 인덱스(tid.y)를
  // gqa_factor로 나눠 대응되는 KV 헤드 인덱스를 구한다.
  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head (Q와 달리 공유 KV 헤드 사용)

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head (Q와 달리 공유 KV 헤드 사용)

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  // Prepare threadgroup memory
  // 아래 값들은 전부 Q/K/V 타일을 threadgroup memory(공유 메모리)에 어떤
  // 레이아웃으로 깔지 정하는 컴파일 타임 상수다.

  // 패딩: 각 행 끝에 덧붙이는 여분 "원소" 수(바이트 아님). 16(=16byte 벡터폭)을
  // sizeof(T)로 나눠 "16바이트를 채우는 T 원소 개수"를 구한다.
  //   half/bf16 (2byte): 16/2 = 8개 원소,  float (4byte): 16/4 = 4개 원소.
  // (8 × 2byte = 16byte 로 딱 맞음 — 8바이트가 아니라 8개라는 뜻.) 목적은 두 가지 -
  //  (1) 16바이트 정렬: 로더/MMA가 128비트 벡터 로드/스토어를 쓰므로 각 행이
  //      16B 경계에서 시작해야 정렬된(빠른) 접근이 된다.
  //  (2) shared memory bank conflict 회피: 행 폭에 여분을 끼워 stride를
  //      어긋나게 해, 같은 컬럼을 읽는 스레드들이 같은 뱅크에 몰리는 직렬화를 줄인다.
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  // Leading dimension(행 stride): 실제 데이터 폭 + 패딩.
  // r행 c열 원소는 smem[r * LD + c]로 접근한다(아래 Qs_offset 등에서 사용).
  // Q: BQ행 x BD(head dim), K: transposed로 적재됨, V: BK행 x BD.
  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  // K 타일과 V 타일은 시간차로 쓰이므로 하나의 버퍼(KV_smem)를 재사용한다.
  // 단 레이아웃이 달라 필요 크기가 다르다:
  //  tgp_mem_0 = K(transposed) 필요량: BD개의 행 x stride(BK+padK)
  //  tgp_mem_1 = V 필요량: BK행 x stride(BD+padV)
  // 한 버퍼에 둘 다 담아야 하니 max로 크기를 잡는다(tgp_mem_s).
  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  // 세 로더 모두 같은 BlockLoaderT 템플릿인데, kDstStrRow/kDstStrCol(SRAM 쪽
  // 행/열 stride)만 바꿔 SRAM에 깔리는 모양을 다르게 만든다.
  //  - BROWS/BCOLS : DRAM에서 읽어올 블록의 행/열 크기
  //  - kDstStrRow/kDstStrCol : SRAM에 쓸 때의 행/열 간격
  //  - reduction_dim : 어느 축을 따라 블록을 흘려보내는가(0=행, 1=열)
  // 자세한 표/그림은 HTML 문서 §5.3 "세 개의 BlockLoader" 참조.
  //***** DRAM -> SRAM *****

  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,           // READ -> DRAM에서 읽을 땐 BQ행 × BD열 (정상 모양)
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp, // WRITE -> 행마다 LDQ_tgp 점프, 열은 1 → row-major 그대로
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  // kDstStrRow=1, kDstStrCol=LDK_tgp 로 행/열 stride를 맞바꿔 전치(transpose)
  // 저장한다. S = Q @ Kᵀ 에 Kᵀ가 필요하므로 적재 시 미리 전치해 두는 트릭. (§5.3)
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,          // READ -> DRAM에서 읽을 땐 BK행 × BD열 (정상 모양)
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,      // WRITE -> ← SRAM에 쓸 땐 행/열 stride를 맞바꿔
      /* short kDstStrCol = */ LDK_tgp,// ← 전치(transpose)해서 저장
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT< // V는 Q와 동일한 row-major
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  // 로더 인스턴스 생성. 두 번째 인자(*_strides[2])는 DRAM 쪽 시퀀스 방향 stride.
  // 위에서 옮겨둔 Q/K/V 시작 포인터(이 TG가 맡은 조각)에서부터 읽기 시작한다.
  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  // scale = 1/sqrt(D). 뒤에서 exp 대신 HW가 빠른 exp2를 쓰므로 log2(e)를 미리
  // 곱해 둔다 (exp(x) = exp2(x·log2(e))). 자세히는 §8 4b 콜아웃 참조.
  const AccumType scale = params->scale * M_LOG2E_F;

  // Prepare MMA tiles
  // Apple GPU의 simdgroup_matrix는 8×8 fragment 단위로 행렬곱-누적(MMA)을 한
  // 명령에 처리한다. 한 SIMD group(32 lane)이 협력해 8×8=64원소를 나눠 갖는다
  // (lane당 2원소). lane→좌표 매핑 그림은 §6 "MMA" 참조.
  constexpr short kFragSize = 8; // MMAFrag size = 8×8
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN; // threadgroup당 SIMD group(warp) 수 = 4
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // 각 축을 8×8 fragment 몇 개로 쪼개는가. 예) BQ=32,BK=16,BD=128,kNWarps=4 →
  // TQ = warp당 Q방향 fragment 수 = 32/(4·8) = 1
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // TK = K방향 fragment 수 = 16/8 = 2 (모든 warp가 같은 K fragment를 본다)
  constexpr int TK = BK / kFragSize;
  // TD = head dim 방향 fragment 수 = 128/8 = 16
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  // register에 사는 행렬 타일들. <AccumType, 행fragment수, 열fragment수, frag>.
  // 코드의 MMATile 객체 의미는 §6 끝의 코드블록에 정리.
  //***** SRAM -> Register *****
  // kTileRows, kTileCols -> 2, 3번째 파라미터
  // MMATile은 공유 변수가 아니라 lane마다 독립된 레지스터 (이름만 같고 위치는 전부 다름).
  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile; // Q 조각 (8쿼리 × 8dim)
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile; // K 조각
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile; // S = Q@Kᵀ (8쿼리 × 16키)
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile; // V 조각
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile; // O 누적 (8쿼리 × 128dim)

  Otile.clear(); // O를 0으로 초기화하고 시작

  // Prepare mma tile offsets
  // 이 lane/warp가 SRAM 타일의 "어느 위치"에서 읽기 시작하는지 계산한다.
  //  - simd_coord = 이 lane이 8×8 fragment 안에서 담당하는 (열x, 행y) 좌표
  //  - sm/sn      = 그 fragment 내 행/열 오프셋 (lane 단위)
  //  - tm         = 이 warp가 맡는 Q행 묶음의 시작 (warp마다 kFragSize씩 떨어짐)
  // → Q는 (행=tm+sm)이 시퀀스, (열=sn)이 dim. K/V는 전치/정상에 따라 다름.
  // 그림과 단계별 설명은 새로 추가한 §6.1 "lane → SRAM 주소 매핑" 참조.
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn; // Q tile 내 내 시작 주소
  const short Ks_offset = sm * LDK_tgp + sn; // K tile(전치) 내 시작 주소
  const short Vs_offset = sm * LDV_tgp + sn; // V tile 내 시작 주소

  // head dim을 8씩 훑을 때 다음 fragment까지의 거리.
  // Q는 dim이 "열"이라 +kFragSize, K는 전치라 dim이 "행"이라 +kFragSize*LDK_tgp.
  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks
  // Q tile은 루프 "밖"에서 단 한 번만 SRAM에 올리고 모든 K 블록에 재사용한다
  // (FlashAttention의 이점). 마지막 Q 블록이 BQ로 안 나눠떨어지면(!align_Q)
  // 경계 밖을 0으로 채우는 load_safe, 꽉 찬 블록은 빠른 load_unsafe. (§8 ②)
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  // Init row reduction variables
  // online softmax 누적기. 각 쿼리 행마다 "지금까지 본 최댓값"과 "지수합"을
  // register에 들고 다니며 블록마다 갱신한다. (수식·이유는 §10)

  // kRowsPerThread
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread; // 이 lane이 맡는 행 수

  AccumType max_score[kRowsPT]; // 행별 running max
  AccumType sum_score[kRowsPT] = {0}; // 행별 running sum(분모)

  // Init to -Inf
  // max를 -inf로 시작해야 첫 블록의 어떤 값이든 max로 채택된다.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  // attention sink: 학습된 가상 토큰 하나를 분모에 미리 더해 두는 기법.
  // max를 sink logit으로, sum을 1로 초기화해 "항상 존재하는 키" 효과를 준다.
  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  // causal(인과) 마스크 최적화: 토큰은 자기보다 미래의 키를 못 본다. 그래서
  // 이 Q 블록이 "볼 필요가 있는" K 블록 범위만 루프 돈다.
  //  kb_lim       = 마지막으로 볼 K 블록(+1). 그 뒤 블록은 전부 미래 → 아예 skip.
  //  kb_min_causal= 부분적으로 미래가 섞이기 시작하는 첫 K 블록(여기부터 마스킹).
  //  qL_off = kL - qL : KV가 Q보다 길 때(캐시된 과거가 있을 때)의 정렬 보정.
  // 예) Q블록0(쿼리0~31)은 K블록0~1만 보면 됨. 그림은 §8 ③, §9 매트릭스.
  int kb_lim = params->NK; // 기본: 전체 K 블록(NK개)을 다 본다
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off; // 이 블록 마지막 쿼리의 절대 위치
    kb_lim = (q_max + BK - 1) / BK; // 그 쿼리가 닿는 마지막 K 블록
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off; // 이 블록 첫 쿼리의 절대 위치
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Loop over KV seq length
  // FlashAttention의 핵심 루프: K/V를 BK 토큰짜리 블록으로 잘라 좌→우로 흘려보내며
  // (streaming) 매 블록마다 [S 계산 → 마스킹 → softmax 갱신 → O 누적]을
  // register/SRAM 안에서 끝낸다. 전체 score 행렬을 통째로 메모리에 두지 않는다.
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K block and apply scale
    // 이번 K 블록을 SRAM(Ks)으로 적재(전치 저장). barrier로 이전 블록의 V 사용이
    // 끝나 KV_smem이 비워졌음을 보장한 뒤 덮어쓴다.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem)); // 마지막 블록: 경계 0채움
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T  (첫 번째 GEMM)
    // head dim(BD=128)을 8씩 TD(=16)조각으로 나눠, 각 dim 조각의 부분곱을
    // Stile에 누적한다 → S[8쿼리 × 16키] 완성. (§8 4a, §9)
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      // SRAM → register fragment 로 이번 dim 조각을 적재
      // LDQ_tgp => BD + padQ
      // LDK_tgp => BK + padK
      // *_offset은 각 lane의 shared_memory 시작점을 알려줌
      // 각 lane이 여러행, 여러열을 맡는다면, (kElemRows, kElemCols -> mma.h)
      // 행을 넘을 때, 4번째 template 파라미터(LDQ_tgp, LDK_tgp)를 사용한다.
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile); // Stile += Qtile · Ktile (MMA)
    }

    // Apply scale in float32
    // S *= 1/sqrt(D)·log2(e). float32(AccumType)로 곱해 정밀도 유지. (§8 4b)
    // kElemsPerTile -> Tile이 몇개인가 * lane당 몇 원소인가
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out length sequence
    // 마지막 K 블록이 BK로 안 나눠떨어질 때, 실재하지 않는 키(kL_rem 초과) 자리를
    // -inf로 막는다. softmax에서 exp2(-inf)=0이 되어 기여하지 않는다.
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      // Stile의 각 fragment 원소를 순회하며, 그 원소가 가리키는 키 열(col_pos)이
      // 실제 키 개수(kL_rem)를 넘으면 -inf.
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    // 같은 블록 안에서도 row_pos(쿼리 절대위치) < col_pos(키 절대위치)면 미래 →
    // -inf. kb_min_causal 이후 블록에서만 검사(그 전 블록은 전부 과거라 통과).
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    // 사용자가 직접 준 mask([B,H,qL,kL]). 두 종류를 지원:
    //  - bool mask: false면 -inf로 막고 true면 통과 (어디를 볼지 on/off)
    //  - additive mask: logit에 값을 더한다(상대 위치 bias 등). exp2 기준이라
    //    여기도 log2(e)를 곱해 더한다.
    // 마스크 시작 포인터는 batch·(Q)head까지만 옮겨져 있어(§7.6) 여기서 시퀀스
    // 좌표(row_pos,col_pos)로 해당 원소를 load_safe로 읽어온다.
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          // 이 fragment에 해당하는 mask 조각을 DRAM에서 읽음(경계 밖은 안전 처리)
          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += M_LOG2E_F * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    // 이제 K 사용이 끝났으니 같은 KV_smem 버퍼를 V로 덮어쓴다(§5.2). barrier가
    // 모든 lane의 S 계산 완료(K 읽기 끝)를 보장한 뒤 적재.
    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax  (online softmax — 블록마다 max·sum을 보정하며 누적, §10)

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i]; // 이전 running max에서 출발
    }

    // Row max: 이번 블록 S까지 포함한 행별 최댓값 (이전 max와 이번 블록 중 큰 값)
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si)): 새 max 기준으로 이번 블록 S를 지수화 (overflow 방지)
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    // max가 갱신됐으므로, 이전까지 누적한 sum·O를 "새 max 기준"으로 재보정할 계수.
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum: 이번 블록의 지수합
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm: 분모 = 이전 합·factor(재보정) + 이번 블록 합
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O: 이전까지 누적한 O도 같은 factor로 재보정 (분모 갱신과 짝을 이룸)
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += S @ V  (두 번째 GEMM, §8 4d / §9)
    // 방금 softmax한 S[8×16(=TK·8)]에 V[16×128]을 곱해 O[8×128]에 누적.
    //  iq: Q행 fragment, id: head dim 출력 fragment, ik: K(키) 방향 reduction.
    // (BD==128일 때만 register 압박이 커 추가 simdgroup_barrier로 안정화)
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize; // V의 키 방향 오프셋
          const short dd = id * kFragSize; // V의 dim 방향 오프셋

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma( // Otile += Stile · Vtile
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration: 로더 포인터를 다음 K/V 블록으로 전진
    loader_k.next();
    loader_v.next();
  }

  // Normalize output
  // 루프 내내 O는 "분자"만 누적했다. 마지막에 한 번 running sum(분모)으로 나눠
  // 진짜 softmax 가중평균을 완성한다. (블록마다 나누지 않는 게 online softmax의 핵심)
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  // register의 O를 DRAM으로 write back. 내 lane이 맡은 (행=tm+sm, 열=sn) 위치로
  // O 포인터를 옮긴 뒤 저장. 포인터 오프셋 의미는 §7 참조.
  O += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    // 마지막 Q 블록(부분 블록): 실제 존재하는 행/열만 안전하게 저장
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return; // 이 lane이 맡을 유효 출력이 없으면 종료

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]); // 꽉 찬 블록: 통째 저장
  }
}
