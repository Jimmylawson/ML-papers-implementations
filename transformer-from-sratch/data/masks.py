import torch

def create_masks(src_tokens, decoder_input, pad_id=1):
    # 1. Source padding mask
    src_mask = (src_tokens == pad_id).unsqueeze(1).unsqueeze(2)
    # 2. Target padding mask
    tgt_padding_mask = (decoder_input == pad_id).unsqueeze(1).unsqueeze(2)
    # 3. Causal mask
    tgt_len = decoder_input.size(1)
    #autoregressive mask
    causal_mask = torch.triu(
        torch.ones(
            tgt_len,
            tgt_len,
            dtype=torch.bool,
            device=decoder_input.device
        ),
        diagonal=1
    )

    # 4. Combine target padding and causal masks
    tgt_mask = (
        causal_mask.unsqueeze(0).unsqueeze(0)
        | tgt_padding_mask
    )

    return src_mask, tgt_mask