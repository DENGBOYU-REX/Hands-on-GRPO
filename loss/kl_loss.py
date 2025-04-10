import torch

# def per_token_kl(ref_per_token_logps, gen_per_token_logps, device = 'cpu'):
#     """_summary_: per_token_kl

#     Args:
#         ref_per_token_logps (_type_): _description_
#         gen_per_token_logps (_type_): _description_
#     """
#     per_token_kl = torch.exp(ref_per_token_logps.to(device) - gen_per_token_logps.to(device)) - (ref_per_token_logps.to(device) - gen_per_token_logps.to(device)) - 1

#     return per_token_kl

def compute_kl_divergence(policy_logprobs, ref_logprobs):
    """计算KL散度，增加数值稳定性"""
    kl_div = torch.exp(ref_logprobs) * (ref_logprobs - policy_logprobs)
    return torch.clamp(kl_div, -100, 100)