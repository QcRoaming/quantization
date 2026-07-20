### 关于bitsandbytes量化：

目前只支持int4和int8量化，可以通过bnb_4bit_quant_type来指定要量化的数据格式，bnb_4bit_compute_dtype来
指定计算时的数据格式，bnb_4bit_compute_dtype一般只支持bfloat16，float32(未指定默认)，float16。int8的量
化有点特殊，其权重中的outlier(异常大值)会按fp16的精度计算，非outlier使用int8，计算完成后转成fp16与outlier
的结果相加。从结果看更像是W4/8A16/32，有时候需要把输入变成bnb_4bit_compute_dtype来指定计算时的数据
格式，这个过程不是量化，因为不产生scale。

性能说明：INT4量化虽然比直接计算BF16多了额外开销（解包，查表，反量化，数据布局转换，专用kernel调度）但也不是一定比bf16慢，因为在一些需要频繁大量从显存往计算单元传输权重的时候，带宽带来的性能提升是可能超过额外开销的。INT8 理论峰值通常高于 BF16，而且权重和激活带宽更低；但 bitsandbytes 的 LLM.int8() 还包含动态量化、outlier 检测/抽取、双路径 GEMM、反量化和结果合并，因此实际速度不一定比直接 BF16 推理快，如果同样模型权重大，输入大，int8计算以及带宽提升的性能也可能超出这些额外开销。

w4a16计算过程源码构建：

```c++
#include <cuda_fp16.h>
#include <cstdint>

// NF4 的 16 项 codebook。
// 真实代码中由 bitsandbytes 内部定义。
// 这里作为参数传入，避免把查找表写死。
template <typename OutputT>
__global__ void dequantize_nf4_kernel(
    const uint8_t* __restrict__ packed_weight, // 每字节保存两个 NF4 权重
    const float* __restrict__ block_absmax,    // 每个量化块对应一个缩放值
    const float* __restrict__ nf4_lut,         // 16 项 NF4 查找表
    OutputT* __restrict__ output,              // FP16/BF16/FP32 输出
    int num_elements,                          // 原始权重元素数
    int block_size                             // 通常是 64 等
) {
    // 一个线程处理一个字节，即两个 4bit 权重。
    const int byte_index =
        blockIdx.x * blockDim.x + threadIdx.x;
const int first_element = byte_index * 2;

if (first_element >= num_elements) {
    return;
}

// 读取一个 packed byte：
//
// bit 7 ........ bit 4 | bit 3 ........ bit 0
//      第一个 NF4 code |      第二个 NF4 code
const uint8_t packed = packed_weight[byte_index];

// 高 4 位。
const uint8_t high_code = packed >> 4;

// 低 4 位。
const uint8_t low_code = packed & 0x0F;

// 找到当前权重所属的量化块。
//
// 同一个 block 内的权重共用一个 absmax。
const int quant_block = first_element / block_size;

const float scale = block_absmax[quant_block];

// NF4 code 本身不是实际权重：
//
// weight ≈ NF4_LUT[code] × block_absmax
const float first_value = nf4_lut[high_code] * scale;
const float second_value = nf4_lut[low_code] * scale;

// 转成目标输出类型，比如 half、bfloat16 或 float。
output[first_element] = static_cast<OutputT>(first_value);

// 奇数个元素时，最后一个 byte 可能只有一个有效 code。
if (first_element + 1 < num_elements) {
    output[first_element + 1] =
        static_cast<OutputT>(second_value);
}
}
```

w8a16计算过程源码构建：

```c++
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// ============================================================
// 精简版 LLM.int8() / Linear8bitLt forward
//
// X: [M, K] FP16 activation
// W: [N, K] FP16 原始权重
//
// Y = X_normal * W_normal^T
//   + X_outlier * W_outlier^T
//
// 普通路径：INT8 × INT8 → INT32 → FP16
// Outlier路径：FP16 × FP16 → FP32
// ============================================================


// ============================================================
// 1. 检测 K 维 outlier 列
//
// 某一列只要出现 |X[m,k]| > threshold，
// 就将整个 K 维标记为 outlier。
// ============================================================

__global__ void detect_outlier_columns(
    const half* __restrict__ X,
    uint8_t* __restrict__ outlier_mask,
    int M,
    int K,
    float threshold
) {
    const int k =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= K) {
        return;
    }

    bool outlier = false;

    for (int m = 0; m < M; ++m) {
        const float value =
            __half2float(X[m * K + k]);

        if (fabsf(value) > threshold) {
            outlier = true;
            break;
        }
    }

    outlier_mask[k] = static_cast<uint8_t>(outlier);
}


// ============================================================
// 2. 根据 outlier mask 生成列索引
//
// 为简化代码，由单线程完成：
// normal_indices  保存普通 K 维
// outlier_indices 保存 outlier K 维
// ============================================================

__global__ void build_column_indices(
    const uint8_t* __restrict__ outlier_mask,
    int* __restrict__ normal_indices,
    int* __restrict__ outlier_indices,
    int* __restrict__ num_normal,
    int* __restrict__ num_outlier,
    int K
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int normal_count = 0;
    int outlier_count = 0;

    for (int k = 0; k < K; ++k) {
        if (outlier_mask[k]) {
            outlier_indices[outlier_count++] = k;
        } else {
            normal_indices[normal_count++] = k;
        }
    }

    *num_normal = normal_count;
    *num_outlier = outlier_count;
}


// ============================================================
// 3. 抽取 activation 的指定 K 维列
//
// input:  [M, K]
// output: [M, selected_K]
// ============================================================

__global__ void gather_activation_columns(
    const half* __restrict__ input,
    const int* __restrict__ indices,
    half* __restrict__ output,
    int M,
    int K,
    int selected_K
) {
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total = M * selected_K;

    if (idx >= total) {
        return;
    }

    const int m = idx / selected_K;
    const int j = idx % selected_K;
    const int original_k = indices[j];

    output[m * selected_K + j] =
        input[m * K + original_k];
}


// ============================================================
// 4. 抽取权重的指定 K 维列
//
// weight: [N, K]
// output: [N, selected_K]
// ============================================================

__global__ void gather_weight_columns(
    const half* __restrict__ weight,
    const int* __restrict__ indices,
    half* __restrict__ output,
    int N,
    int K,
    int selected_K
) {
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total = N * selected_K;

    if (idx >= total) {
        return;
    }

    const int n = idx / selected_K;
    const int j = idx % selected_K;
    const int original_k = indices[j];

    output[n * selected_K + j] =
        weight[n * K + original_k];
}


// ============================================================
// 5. Activation 按行动态量化为 INT8
//
// X_q[m,k] = round(X[m,k] / scale_x[m])
// scale_x[m] = absmax(X[m,:]) / 127
// ============================================================

__global__ void quantize_activation_rowwise(
    const half* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int M,
    int K
) {
    const int row = blockIdx.x;

    if (row >= M) {
        return;
    }

    extern __shared__ float shared_max[];

    float local_max = 0.0f;

    for (int k = threadIdx.x;
         k < K;
         k += blockDim.x) {

        const float value =
            __half2float(input[row * K + k]);

        local_max =
            fmaxf(local_max, fabsf(value));
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    // 求当前 activation 行的 absmax。
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] =
                fmaxf(
                    shared_max[threadIdx.x],
                    shared_max[threadIdx.x + stride]
                );
        }

        __syncthreads();
    }

    const float absmax = shared_max[0];

    // 保存反量化时使用的 scale。
    const float dequant_scale =
        absmax > 0.0f
            ? absmax / 127.0f
            : 0.0f;

    if (threadIdx.x == 0) {
        scales[row] = dequant_scale;
    }

    __syncthreads();

    const float quant_scale =
        absmax > 0.0f
            ? 127.0f / absmax
            : 0.0f;

    for (int k = threadIdx.x;
         k < K;
         k += blockDim.x) {

        const float value =
            __half2float(input[row * K + k]);

        int q =
            __float2int_rn(value * quant_scale);

        q = max(-127, min(127, q));

        output[row * K + k] =
            static_cast<int8_t>(q);
    }
}


// ============================================================
// 6. 权重按输出通道量化为 INT8
//
// W_q[n,k] = round(W[n,k] / scale_w[n])
// scale_w[n] = absmax(W[n,:]) / 127
//
// 实际推理中通常在模型加载阶段提前完成。
// ============================================================

__global__ void quantize_weight_rowwise(
    const half* __restrict__ weight,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int N,
    int K
) {
    const int row = blockIdx.x;

    if (row >= N) {
        return;
    }

    extern __shared__ float shared_max[];

    float local_max = 0.0f;

    for (int k = threadIdx.x;
         k < K;
         k += blockDim.x) {

        const float value =
            __half2float(weight[row * K + k]);

        local_max =
            fmaxf(local_max, fabsf(value));
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] =
                fmaxf(
                    shared_max[threadIdx.x],
                    shared_max[threadIdx.x + stride]
                );
        }

        __syncthreads();
    }

    const float absmax = shared_max[0];

    const float dequant_scale =
        absmax > 0.0f
            ? absmax / 127.0f
            : 0.0f;

    if (threadIdx.x == 0) {
        scales[row] = dequant_scale;
    }

    __syncthreads();

    const float quant_scale =
        absmax > 0.0f
            ? 127.0f / absmax
            : 0.0f;

    for (int k = threadIdx.x;
         k < K;
         k += blockDim.x) {

        const float value =
            __half2float(weight[row * K + k]);

        int q =
            __float2int_rn(value * quant_scale);

        q = max(-127, min(127, q));

        output[row * K + k] =
            static_cast<int8_t>(q);
    }
}


// ============================================================
// 7. 普通路径：INT8 × INT8 → INT32
//
// X_q: [M, K_normal]
// W_q: [N, K_normal]
// C:   [M, N]
//
// C[m,n] = Σ X_q[m,k] * W_q[n,k]
// ============================================================

__global__ void int8_matmul(
    const int8_t* __restrict__ X,
    const int8_t* __restrict__ W,
    int32_t* __restrict__ accumulator,
    int M,
    int N,
    int K
) {
    const int n =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int m =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) {
        return;
    }

    int32_t acc = 0;

    for (int k = 0; k < K; ++k) {
        const int32_t x =
            static_cast<int32_t>(
                X[m * K + k]
            );

        const int32_t w =
            static_cast<int32_t>(
                W[n * K + k]
            );

        // INT8 乘法，INT32 累加。
        acc += x * w;
    }

    accumulator[m * N + n] = acc;
}


// ============================================================
// 8. INT32 结果反量化为浮点
//
// Y_normal[m,n] =
//     C_int32[m,n]
//     * scale_x[m]
//     * scale_w[n]
// ============================================================

__global__ void dequantize_int32_result(
    const int32_t* __restrict__ accumulator,
    const float* __restrict__ activation_scales,
    const float* __restrict__ weight_scales,
    float* __restrict__ output,
    int M,
    int N
) {
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total = M * N;

    if (idx >= total) {
        return;
    }

    const int m = idx / N;
    const int n = idx % N;

    output[idx] =
        static_cast<float>(accumulator[idx])
        * activation_scales[m]
        * weight_scales[n];
}


// ============================================================
// 9. Outlier 路径：FP16 × FP16 → FP32
//
// X_outlier: [M, K_outlier]
// W_outlier: [N, K_outlier]
// ============================================================

__global__ void outlier_fp16_matmul(
    const half* __restrict__ X,
    const half* __restrict__ W,
    float* __restrict__ output,
    int M,
    int N,
    int K_outlier
) {
    const int n =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int m =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) {
        return;
    }

    float acc = 0.0f;

    for (int k = 0; k < K_outlier; ++k) {
        const float x =
            __half2float(
                X[m * K_outlier + k]
            );

        const float w =
            __half2float(
                W[n * K_outlier + k]
            );

        acc += x * w;
    }

    output[m * N + n] = acc;
}


// ============================================================
// 10. 合并普通 INT8 路径、FP16 outlier 路径和 bias
//
// Y = Y_normal + Y_outlier + bias
// ============================================================

__global__ void combine_results(
    const float* __restrict__ normal_result,
    const float* __restrict__ outlier_result,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int M,
    int N
) {
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total = M * N;

    if (idx >= total) {
        return;
    }

    const int n = idx % N;

    float value =
        normal_result[idx]
        + outlier_result[idx];

    if (bias != nullptr) {
        value += __half2float(bias[n]);
    }

    output[idx] = __float2half(value);
}
```

### GPTQ量化过程：

离线量化：GPTQ使用一小批校准数据，让模型运行若干次，记录各个 Linear 层的输入，它利用这些输入估计：哪些输入通道更重要，哪些权重列之间相关，某个权重量化产生的误差会怎样影响输出，然后逐列量化，首先计算当前列量化误差，然后把误差补偿（Hessian）到未量化列，然后继续量化下一列。量化后的 Linear 层通常保存四类数据：qweight：低比特整数权重，通常打包到INT32中，scales：每组权重的缩放因子，qzeros：每组权重的零点，g_idx：每个输入通道使用哪一组scale和zero。在torch后端，也就是GPTQModel 的 Torch 路径在反量化后会把权重转换到输入 `x.dtype`(激活值类型：经过 Embedding 层以后产生的浮点 `hidden_states`的数据类型，而这个类型可以由模型初始化阶段指定，如果要对GPTQ量化后的模型推理，模型初始化阶段一般指定fp16/bf16，如果指定fp32根据不同的后端，要么报错，要么自动转换成fp16 )，随后调用 `torch.matmul`。之后的源码整理也是torch后端。每个后端执行矩阵乘的时候权重都会反量化回 x.dtype

torch后端源码：

```python
# 1. 从打包的 int32 中解出 INT4 权重
weight_int = (qweight >> shifts) & 0xF

# 2. 解出每个量化组对应的 zero
zeros = (qzeros >> shifts) & 0xF

# 3. 根据 g_idx，为每个输入通道选择对应的 scale 和 zero
group_ids = g_idx.long()

scale = scales[group_ids]
zero = zeros[group_ids]

# 4. 反量化
# 常见做法：先用 FP32 计算，再转换为 activation 的 dtype
weight = scale.float() * (
    weight_int.float() - zero.float()
)

weight = weight.to(x.dtype)

# 5. 矩阵乘
output = x @ weight

# 6. 加 bias
if bias is not None:
    output = output + bias.to(output.dtype)
```

GPTQ量化中常用参数：

- **bits**
  指定权重量化位数，例如 `4` 表示将权重量化为 INT4。
- **group_size**
  指定多少个连续权重共享一组 `scale` 和 `zero-point`。数值越小，量化粒度越细，但量化参数更多。
- **desc_act**
  是否根据激活重要程度重新排列权重列的量化顺序。开启后通常有利于精度，但可能影响推理速度和后端兼容性。
- **true_sequential**
  是否按照模型真实执行顺序逐层量化。开启后，后面的层会使用前面已经量化后的输出，更接近最终推理状态。
- **sym**
  是否采用对称量化。`True` 表示量化范围以 0 为中心；`False` 表示使用带 zero-point 的非对称量化。
- **damp_percent**
  对 GPTQ 使用的 Hessian 矩阵增加阻尼，避免矩阵求逆或分解时数值不稳定。
- **dataset**
  指定校准数据。GPTQ使用这些数据收集模型各层的输入激活，并据此判断不同权重的量化敏感度。

最常见的配置形式大致是：

```python
GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    true_sequential=True,
    sym=True,
    damp_percent=0.1,
    dataset=calibration_data,
)
```

