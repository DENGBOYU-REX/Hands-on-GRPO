# import torch

# def generate_probs_mask(model, tokenizer, inputs:list, max_length:int=512, device='cuda'):

#     model.to(device)
#     inputs = tokenizer(
#         inputs,
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length
#     ).to(device)

#     # 使用模型生成 logits
#     with torch.no_grad():
#         outputs = model(inputs.input_ids)  # (batch_size, input_length, vocab_size)

#     # 获取 logits
#     logits = outputs.logits  # (batch_size, input_length, vocab_size)

#     # 将 logits 转换为概率分布
#     probs = torch.softmax(logits, dim=-1)  # (batch_size, input_length, vocab_size)

#     return probs, inputs['attention_mask']

import torch
import asyncio

async def generate_text_with_probs(model, tokenizer, inputs: list, max_length: int = 512, device='cuda'):
    model.to(device)
    # print (model)
    
    # 编码输入文本
    inputs = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    # 自回归生成文本并保留概率
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    # 提取生成的token IDs和概率
    generated_ids = outputs.sequences  # (batch_size, seq_len)
    scores = outputs.scores
    probs = torch.stack([torch.softmax(score, dim=-1) for score in scores], dim=1)  # (batch_size, gen_steps, vocab_size)
    log_probs = torch.stack([torch.log_softmax(score, dim=-1).clamp(min=-1e20) for score in scores], dim=1)

    # # 创建位置索引矩阵 [1, 2, 3, ..., max_length]
    # position_indices = torch.arange(max_length, device=device).expand(generated_ids.size(0), -1)
    
    # 生成completion_mask（标记生成部分的非填充token）
    completion_mask = (generated_ids != tokenizer.pad_token_id).int()  # (batch_size, seq_len)

    # 计算实际输入长度（排除padding）
    input_lengths = inputs.attention_mask.sum(dim=1)  # (batch_size,)
    # 解码文本
    generated_texts = tokenizer.batch_decode(generated_ids[:, input_lengths.unsqueeze(1):], skip_special_tokens=True)
    # 计算实际生成长度
    # gen_lengths = (generated_ids != tokenizer.pad_token_id).sum(dim=1)

    # 确保 probs 和 completion_mask 的长度为 max_length
    batch_size = generated_ids.size(0)
    padded_probs = torch.zeros(batch_size, max_length, probs.size(-1), device=device)
    padded_log_probs = torch.zeros(batch_size, max_length, log_probs.size(-1), device=device)
    padded_completion_mask = torch.zeros(batch_size, max_length, dtype=torch.int, device=device)

    for i in range(batch_size):
        gen_len = min(len(scores), max_length)
        padded_probs[i, :gen_len] = probs[i, :gen_len]
        padded_completion_mask[i, :gen_len] = completion_mask[i, :gen_len]
        padded_log_probs[i, :gen_len] = log_probs[i, :gen_len]

    return {
        "texts": generated_texts,
        "probs": padded_probs,
        "log_probs": padded_log_probs,
        "completion_mask": padded_completion_mask,
        "generated_ids": generated_ids,
     }