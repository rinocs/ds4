// Qwen3 Gated Attention and Recurrent Gated DeltaNet Linear Attention Kernels

kernel void kernel_qwen3_gated_attention(
    device const float * q            [[buffer(0)]],
    device const float * k            [[buffer(1)]],
    device const float * v            [[buffer(2)]],
    device const float * gate         [[buffer(3)]],
    device       float * dst          [[buffer(4)]],
    constant     uint  & n_tokens     [[buffer(5)]],
    constant     uint  & n_heads      [[buffer(6)]],
    constant     uint  & head_dim     [[buffer(7)]],
    uint3                tgpig        [[threadgroup_position_in_grid]],
    uint3                tpitg        [[thread_position_in_threadgroup]],
    uint3                ntg          [[threads_per_threadgroup]]
) {
    uint token_idx = tgpig.x;
    uint head_idx = tgpig.y;
    
    if (token_idx >= n_tokens || head_idx >= n_heads) return;
    
    uint offset = token_idx * n_heads * head_dim + head_idx * head_dim;
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        float q_val = q[offset + i];
        float k_val = k[offset + i];
        float v_val = v[offset + i];
        float g_val = gate[offset + i];
        
        // Sigmoid gate
        float gate_sig = 1.0f / (1.0f + exp(-g_val));
        
        dst[offset + i] = q_val * k_val * v_val * gate_sig;
    }
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
    uint token_idx = tgpig.x;
    uint head_idx = tgpig.y;
    
    if (token_idx >= n_tokens || head_idx >= n_heads) return;
    
    uint offset = token_idx * n_heads * head_dim + head_idx * head_dim;
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        float q_val = q[offset + i];
        float k_val = k[offset + i];
        float v_val = v[offset + i];
        float b_val = beta[offset + i];
        
        float current_state = state[offset + i];
        float new_state = current_state + b_val * (v_val - k_val * current_state);
        state[offset + i] = new_state;
        
        dst[offset + i] = q_val * new_state;
    }
}
