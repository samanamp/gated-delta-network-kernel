"""
Triton Kernel Template for FlashInfer Competition.

Implement your kernel logic here. The entry point function name should match
the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

import triton
import triton.language as tl


@triton.jit
def gdn_kernel(
    state_ptr, state_out_ptr, out_ptr,
    q_ptr, k_ptr, v_ptr,
    a_ptr, dt_bias_ptr, alog_ptr, b_ptr,
    # ... strides ...
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,
    stride_state_out_b, stride_state_out_h, stride_state_out_v, stride_state_out_k,
    stride_a_b, stride_a_h,
    stride_dt_bias,
    stride_alog,
    stride_b_b, stride_b_h,
    stride_q_b, stride_q_h, stride_q_k,
    stride_k_b, stride_k_h, stride_k_k,
    stride_v_b, stride_v_h, stride_v_v,
    stride_out_b, stride_out_h, stride_out_v,
    V, K: tl.constexpr, 
    BLOCK_V: tl.constexpr,
    Hv: tl.constexpr,
    Hq: tl.constexpr,
    # scale_ptr
    scale: tl.constexpr,
):

    b = tl.program_id(0)
    h = tl.program_id(1)
    v_block = tl.program_id(2)
    # scale = tl.load(scale_ptr)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_k = tl.arange(0, K)

    # x = a.float() + dt_bias.view(1, 1, Hv).float()  # [B,1,Hv]
    # a [b, 1, hv]
    a_data_ptr = a_ptr + b*stride_a_b + h*stride_a_h
    a_data = tl.load(a_data_ptr)

    dt_bias_data_ptr = dt_bias_ptr + h*stride_dt_bias
    dt_bias_data = tl.load(dt_bias_data_ptr)

    x = a_data + dt_bias_data
    # g = exp(-exp(A_log)+Softplus(x))
    alog_data_ptr = alog_ptr + h*stride_alog
    alog_data = tl.load(alog_data_ptr)
    g = (tl.exp(-tl.exp(alog_data.to(tl.float32)) * tl.log(1.0 + tl.exp(x.to(tl.float32)))))
    # softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    # g = tl.exp(-tl.exp(alog_data) * softplus_x)

    # beta = sigmoid (b)
    b_data_ptr = b_ptr + b * stride_b_b + h * stride_b_h
    b_data = tl.load(b_data_ptr).to(tl.float32)
    beta = tl.sigmoid(b_data)
    
    # pointer to state [b, h, offs_v, offs_k] - a [BLOCK_V, K] tile
    ptrs = (state_ptr
            + b * stride_state_b
            + h * stride_state_h
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
           )

    state_tile = tl.load(ptrs)

    # Apply gate to the state tile
    state_tile = g * state_tile

    rep = Hv // Hq
    k_head = h // rep
    k_data_ptr = k_ptr + b * stride_k_b + k_head * stride_k_h + offs_k * stride_k_k
    k_data_vector = tl.load(k_data_ptr).to(tl.float32)    # shape: [K]
    
    # q_head = h // rep
    q_data_ptr = q_ptr + b * stride_q_b + k_head * stride_q_h + offs_k * stride_q_k
    q_data_vector = tl.load(q_data_ptr).to(tl.float32)   # shape: [K]
    
    # old_v = k @ old_state.trans  -> [B,Hv,V]
    old_v = tl.sum(k_data_vector[None, :] * state_tile, axis=1)  # [BLOCK_V]
    
    # v_data for this head and batch
    v_data_ptr = v_ptr + b * stride_v_b + h * stride_v_h + offs_v * stride_v_v
    mask_v = offs_v < V
    v_data_vector = tl.load(v_data_ptr, mask=mask_v, other=0.0).to(tl.float32)   # shape: [BLOCK_V]
    
    # new_v = beta * v + (1-beta) * old_v
    new_v = beta * v_data_vector + (1.0 - beta) * old_v
    
    # delta_v = new_v - old_v
    delta_v = new_v - old_v

    # Update state: state += k^T @ delta_v
    state_tile += k_data_vector[None, :] * delta_v[:, None]
    
    

    # Compute output: scale * q @ state_new
    output = tl.sum(q_data_vector[None, :] * state_tile, axis=1)  # [BLOCK_V]
    
    # Store output
    out_data_ptr = out_ptr + b * stride_out_b + h * stride_out_h + offs_v * stride_out_v
    tl.store(out_data_ptr, (scale * output).to(tl.bfloat16), mask=mask_v)

    #write it back (same layout)
    new_state_ptrs = (state_out_ptr
               + b * stride_state_out_b
               + h * stride_state_out_h
               + offs_v[:, None] * stride_state_out_v
               + offs_k[None, :] * stride_state_out_k
                )
    mask_state = mask_v[:, None]   # Broadcast mask_v to [BLOCK_V, 1]
    tl.store(new_state_ptrs, state_tile, mask=mask_state)

# def tritonGDN(*args, **kwargs):
#     print("Positional arguments:")
#     for i, arg in enumerate(args):
#         print(f"  arg[{i}] = {arg}")

#     print("\nKeyword arguments:")
#     for key, value in kwargs.items():
#         print(f"  {key} = {value}")
def tritonGDN(q, k, v, state, A_log, a, dt_bias, b, scale, out, state_out):
    B, T, Hq, K = q.shape
    _, _, Hk, _ = k.shape
    _, _, Hv, V = v.shape
    # assert T == 1
    # assert Hk == Hq
    # assert Hv % Hq == 0
    # if state is None:
    #     state = torch.zeros((B, Hv, V, K), device=q.device, dtype=torch.float32)
    # else:
    #     state = state.float()
    
    # if scale is None or float(scale) == 0.0:
    #     scale = 1.0 / math.sqrt(K)
    
    # Create output tensors
    # out = torch.empty((B, T, Hv, V), device=q.device, dtype=q.dtype)
    # state_out = torch.empty_like(state)
    
    # Launch triton kernel
    BLOCK_V = min(8, triton.next_power_of_2(V))
    grid = (B, Hv, triton.cdiv(V, BLOCK_V))
    
    gdn_kernel[grid](
        state, state_out, out,
        q, k, v,
        a, dt_bias, A_log, b,
        # State strides [B, Hv, V, K]
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # State_out strides [B, Hv, V, K]
        state_out.stride(0), state_out.stride(1), state_out.stride(2), state_out.stride(3),
        # a strides [B, T, Hv] -> need stride(0), stride(2) for [B, Hv]
        a.stride(0), a.stride(2),
        # dt_bias stride [Hv]
        dt_bias.stride(0),
        # A_log stride [Hv]
        A_log.stride(0),
        # b strides [B, T, Hv] -> need stride(0), stride(2) for [B, Hv]
        b.stride(0), b.stride(2),
        # q strides [B, T, Hq, K] -> need stride(0), stride(2), stride(3) for [B, Hq, K]
        q.stride(0), q.stride(2), q.stride(3),
        # k strides [B, T, Hk, K] -> need stride(0), stride(2), stride(3) for [B, Hk, K]
        k.stride(0), k.stride(2), k.stride(3),
        # v strides [B, T, Hv, V] -> need stride(0), stride(2), stride(3) for [B, Hv, V]
        v.stride(0), v.stride(2), v.stride(3),
        # out strides [B, T, Hv, V] -> need stride(0), stride(2), stride(3) for [B, Hv, V]
        out.stride(0), out.stride(2), out.stride(3),
        # Constexprs
        V=V, K=K, BLOCK_V=BLOCK_V,
        Hv=Hv, Hq=Hq, scale=scale
    )
    
    return out, state_out
