// Qwen3 Gated Attention and Recurrent Gated DeltaNet Linear Attention Kernels

kernel void kernel_qwen3_gated_attention(
    device const float * src          [[buffer(0)]],
    device const float * gate         [[buffer(1)]],
    device       float * dst          [[buffer(2)]],
    constant     uint  & n_elements   [[buffer(3)]],
    uint                 gid          [[thread_position_in_grid]]
) {
    if (gid >= n_elements) return;
    float g_val = gate[gid];
    float gate_sig = 1.0f / (1.0f + exp(-g_val));
    dst[gid] = src[gid] * gate_sig;
}

kernel void kernel_qwen3_deltanet(
    device const float * q            [[buffer(0)]],
    device const float * k            [[buffer(1)]],
    device const float * v            [[buffer(2)]],
    device const float * beta         [[buffer(3)]],
    device       float * state        [[buffer(4)]], // recurrent state tensor
    device       float * dst          [[buffer(5)]],
    constant     uint  & n_tokens     [[buffer(6)]],
    constant     uint  & n_heads      [[buffer(7)]],
    constant     uint  & head_dim     [[buffer(8)]],
    uint3                tgpig        [[threadgroup_position_in_grid]],
    uint3                tpitg        [[thread_position_in_threadgroup]],
    uint3                ntg          [[threads_per_threadgroup]]
) {
    uint head_idx = tgpig.x;
    if (head_idx >= n_heads) return;

    threadgroup float s_shared[128 * 128];
    threadgroup float u_shared[128];
    threadgroup float k_shared[128];
    threadgroup float v_shared[128];
    threadgroup float q_shared[128];
    threadgroup float beta_shared[128];

    uint state_offset = head_idx * head_dim * head_dim;
    uint total_elements = head_dim * head_dim;
    
    // Load initial state S from global state buffer
    for (uint idx = tpitg.x; idx < total_elements; idx += ntg.x) {
        s_shared[idx] = state[state_offset + idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < n_tokens; ++t) {
        uint token_offset = t * n_heads * head_dim + head_idx * head_dim;
        
        // Cooperatively load vectors
        for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
            q_shared[i] = q[token_offset + i];
            k_shared[i] = k[token_offset + i];
            v_shared[i] = v[token_offset + i];
            beta_shared[i] = beta[token_offset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute u_t = S k_t
        for (uint r = tpitg.x; r < head_dim; r += ntg.x) {
            float sum = 0.0f;
            for (uint c = 0; c < head_dim; ++c) {
                sum += s_shared[r * head_dim + c] * k_shared[c];
            }
            u_shared[r] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S_t = S_{t-1} + beta_t (v_t - u_t) k_t^T
        for (uint idx = tpitg.x; idx < total_elements; idx += ntg.x) {
            uint r = idx / head_dim;
            uint c = idx % head_dim;
            s_shared[idx] += beta_shared[r] * (v_shared[r] - u_shared[r]) * k_shared[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute output o_t = S_t q_t
        for (uint r = tpitg.x; r < head_dim; r += ntg.x) {
            float sum = 0.0f;
            for (uint c = 0; c < head_dim; ++c) {
                sum += s_shared[r * head_dim + c] * q_shared[c];
            }
            dst[token_offset + r] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final state back
    for (uint idx = tpitg.x; idx < total_elements; idx += ntg.x) {
        state[state_offset + idx] = s_shared[idx];
    }
}
