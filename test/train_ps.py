from utils.policy_model import policy_model
from utils.generate_probs_mask import generate_text_with_probs
from utils.ref_model import ref_model
from utils.reward_func import dummy_reward_fn
from loss.kl_loss import per_token_kl
import torch
import asyncio

# class Trainer:
# # 单个数据处理
#     def __init__(self, model_path, ref_model_path, ref_model_device = 'cuda:0', policy_model_device = 'cuda:1', reward_fn=dummy_reward_fn):
#         self.policy_model, self.tokenizer = policy_model(model_path, device=policy_model_device)
#         self.ref_model, _ = ref_model(ref_model_path, device=ref_model_device)
#         self.reward_fn = reward_fn
#         self.ref_model_device = ref_model_device
#         self.policy_model_device = policy_model_device
    
#     async def train(self, input_data, n_generates=4, beta=0.01, clip_param = 0.2):
#         # # TODO: Implement training loop
#         # pass
#         self.policy_model.train()
#         self.ref_model.eval()

#         policy_model_result_all = []
#         ref_model_result_all = []

#         async def generate_single(i):
#             """
#             单次生成任务。
#             """
#             policy_model_result = await generate_text_with_probs(
#                 model=self.policy_model,
#                 tokenizer=self.tokenizer,
#                 inputs=input_data,
#                 device=self.policy_model_device
#             )
#             ref_model_result = await generate_text_with_probs(
#                 model=self.ref_model,
#                 tokenizer=self.tokenizer,
#                 inputs=input_data,
#                 device=self.ref_model_device
#             )
#             policy_model_result_all.append(policy_model_result)
#             ref_model_result_all.append(ref_model_result)

#         # 并行运行所有生成任务
#         tasks = [generate_single(i) for i in range(n_generates)]
#         await asyncio.gather(*tasks)

#         # 堆叠生成结果
#         policy_model_results_stacked_probs = torch.cat([result['probs'] for result in policy_model_result_all], dim=0)
#         # ref_model_results_stacked_probs = {key: torch.stack([result[key] for result in ref_model_result_all]) for key in ref_model_result_all[0]}

#         print ("policy_model_results_stacked['probs'].shape:", policy_model_results_stacked_probs.shape)

#         kl_loss_all = []
#         for ref_probs, policy_probs in zip(ref_model_result_all, policy_model_result_all):
#             per_token_kl_ = per_token_kl(ref_probs['probs'] * ref_probs['completion_mask'].unsqueeze(-1), policy_probs['probs'] * policy_probs['completion_mask'].unsqueeze(-1))
#             # kl_loss_all.append(sum(kl_loss_))
#             kl_loss_all.append(torch.sum(per_token_kl_), dim=[0, 1])
#             # print ('kl_loss_:', kl_loss_)
        
#         rewards_all = []
#         advantages_all = []
#         for ref_answers, policy_answers in zip(ref_model_result_all, policy_model_result_all):
#             # print ('ref_answers:', ref_answers['texts'])
#             # print ('policy_answers:', policy_answers['texts'])
#             print (policy_answers['texts'])
#             reward = dummy_reward_fn(policy_answers['texts'][0])
#             rewards_all.append(reward)
#             advantages_all.append(reward)

#             # print ('reward:', reward)

#         # 组计算：平均奖励和优势值
#         group_rewards = sum(rewards_all) / len(rewards_all)
#         advantages = torch.tensor(advantages_all).to('cpu')
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#         print ('advantages:', advantages)
#         # 计算 PPO 损失
#         ppo_loss = []
#         for ref_probs, policy_probs, adv in zip(ref_model_result_all, policy_model_result_all, advantages):
#             # 提取 log_probabilities
#             policy_log_probs = torch.log(policy_probs['probs'] + 1e-8)  # 防止数值溢出
#             ref_log_probs = torch.log(ref_probs['probs'] + 1e-8)
#             print('ref_log_probs:', ref_log_probs.shape)

#             # 计算比率
#             ratio = torch.exp(policy_log_probs.to('cpu') - ref_log_probs.detach().to('cpu'))
            
#             # 处理序列长度维度上的比率
#             if len(ratio.shape) == 3:  # 如果有 vocab 维度，应该去掉它
#                 ratio = ratio.mean(dim=-1)  # 对词汇表维度求均值
            
#             clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)

#             # 计算 PPO 损失
#             surr1 = ratio * adv
#             surr2 = clipped_ratio * adv
#             ppo_loss.append(-torch.mean(torch.min(surr1, surr2)))
#         print ('ppo_loss:', ppo_loss)

#         # # 组计算
#         # group_rewards = sum(rewards_all) / len(rewards_all)

#         # print (kl_loss_all)
#         # # 更新策略模型
#         # optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-5)
#         # total_loss = -group_rewards + beta * sum(kl_loss_all)
#         # print (total_loss)
#         return policy_model_result_all, ref_model_result_all

# class Trainer:
# # 单个数据处理
#     def __init__(self, model_path, ref_model_path, ref_model_device = 'cuda:0', policy_model_device = 'cuda:1', reward_fn=dummy_reward_fn):
#         self.policy_model, self.tokenizer = policy_model(model_path, device=policy_model_device)
#         self.ref_model, _ = ref_model(ref_model_path, device=ref_model_device)
#         self.reward_fn = reward_fn
#         self.ref_model_device = ref_model_device
#         self.policy_model_device = policy_model_device
    
#     async def train(self, input_data, n_generates=4, beta=0.01, clip_param = 0.2):
#         # # TODO: Implement training loop
#         # pass
#         self.policy_model.train()
#         self.ref_model.eval()

#         policy_model_result_all = []
#         ref_model_result_all = []

#         async def generate_single(i):
#             """
#             单次生成任务。
#             """
#             policy_model_result = await generate_text_with_probs(
#                 model=self.policy_model,
#                 tokenizer=self.tokenizer,
#                 inputs=input_data,
#                 device=self.policy_model_device
#             )
#             ref_model_result = await generate_text_with_probs(
#                 model=self.ref_model,
#                 tokenizer=self.tokenizer,
#                 inputs=input_data,
#                 device=self.ref_model_device
#             )
#             policy_model_result_all.append(policy_model_result)
#             ref_model_result_all.append(ref_model_result)

#         # 并行运行所有生成任务
#         tasks = [generate_single(i) for i in range(n_generates)]
#         await asyncio.gather(*tasks)

#         # 堆叠生成结果
#         policy_model_results_stacked_log_probs = torch.cat([result['log_probs'] for result in policy_model_result_all], dim=0)
#         ref_model_results_stacked_log_probs = torch.cat([result['log_probs'] for result in ref_model_result_all], dim=0)

#         print ('ref_model_results_stacked_log_probs',ref_model_results_stacked_log_probs)
#         print ('ref_model_results_stacked_log_probs',ref_model_results_stacked_log_probs.shape)

#         policy_model_results_stacked_mask = torch.cat([result['completion_mask'] for result in policy_model_result_all], dim=0)
#         print ('policy_model_results_stacked_mask.shape:', policy_model_results_stacked_mask.shape)

#         per_token_kl_ = per_token_kl(policy_model_results_stacked_log_probs, ref_model_results_stacked_log_probs)
#         print(per_token_kl_.shape)


#         ##ration
#         ratio = torch.exp(policy_model_results_stacked_log_probs.to('cpu') - ref_model_results_stacked_log_probs.to('cpu'))

#         clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)

        
#         print(ratio.shape)
#         rewards_all = []
#         advantages_all = []
#         for ref_answers, policy_answers in zip(ref_model_result_all, policy_model_result_all):
#             # print ('ref_answers:', ref_answers['texts'])
#             # print ('policy_answers:', policy_answers['texts'])
#             print (policy_answers['texts'])
#             reward = dummy_reward_fn(policy_answers['texts'][0])
#             rewards_all.append(reward)
#             advantages_all.append(reward)

#             # print ('reward:', reward)

#         # 组计算：平均奖励和优势值
#         group_rewards = sum(rewards_all) / len(rewards_all)
#         advantages = torch.tensor(advantages_all).to('cpu')
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#         advantages = advantages.view(n_generates, 1, 1)
#         print ('advantages:', advantages)

#         per_token_loss_ = torch.min(ratio * advantages, clipped_ratio * advantages)

#         print (per_token_loss_.shape)

#         per_token_loss_ = - (per_token_loss_ - beta * per_token_kl_)

#         tmp = per_token_loss_.to('cpu') * policy_model_results_stacked_mask.unsqueeze(-1).to('cpu')

#         # 对每个批次的掩码求和
#         mask_sum = policy_model_results_stacked_mask.sum(dim=1).unsqueeze(-1)  # 形状为 [2, 1]
#         print (mask_sum)
#         # loss = (tmp.sum(dim=1).to('cpu') / mask_sum.to('cpu')).mean()
#         mask_sum = policy_model_results_stacked_mask.sum(dim=1).unsqueeze(-1)
#         loss = (tmp.sum(dim=1).to('cpu') / mask_sum.to('cpu')).mean()
#         print (tmp.sum())
#         print(loss)
#         # per_token_loss = -(per_token_loss - beta * per_token_kl)

#         # # 计算 PPO 损失
#         # ppo_loss = []
#         # for ref_probs, policy_probs, adv in zip(ref_model_result_all, policy_model_result_all, advantages):
#         #     # 提取 log_probabilities
#         #     policy_log_probs = torch.log(policy_probs['probs'] + 1e-8)  # 防止数值溢出
#         #     ref_log_probs = torch.log(ref_probs['probs'] + 1e-8)
#         #     print('ref_log_probs:', ref_log_probs.shape)

#         #     # 计算比率
#         #     ratio = torch.exp(policy_log_probs.to('cpu') - ref_log_probs.detach().to('cpu'))
            
#         #     # 处理序列长度维度上的比率
#         #     if len(ratio.shape) == 3:  # 如果有 vocab 维度，应该去掉它
#         #         ratio = ratio.mean(dim=-1)  # 对词汇表维度求均值
            
#         #     clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)

#         #     # 计算 PPO 损失
#         #     surr1 = ratio * adv
#         #     surr2 = clipped_ratio * adv
#         #     ppo_loss.append(-torch.mean(torch.min(surr1, surr2)))
#         # print ('ppo_loss:', ppo_loss)

#         # # # 组计算
#         # # group_rewards = sum(rewards_all) / len(rewards_all)

#         # # print (kl_loss_all)
#         # # # 更新策略模型
#         # # optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=1e-5)
#         # # total_loss = -group_rewards + beta * sum(kl_loss_all)
#         # # print (total_loss)
#         return policy_model_result_all, ref_model_result_all


class Trainer:
    def __init__(self, model_path, ref_model_path, ref_model_device='cuda:0', policy_model_device='cuda:1', reward_fn=dummy_reward_fn):
        self.policy_model, self.tokenizer = policy_model(model_path, device=policy_model_device)
        self.ref_model, _ = ref_model(ref_model_path, device=ref_model_device)
        self.reward_fn = reward_fn
        self.ref_model_device = ref_model_device
        self.policy_model_device = policy_model_device

    def pad_sequences(self, results, max_len):
        padded_ids = torch.full((len(results), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_mask = torch.zeros(len(results), max_len, dtype=torch.int)
        
        for i, result in enumerate(results):
            seq_len = min(len(result['generated_ids'][0]), max_len)
            padded_ids[i, :seq_len] = result['generated_ids'][0][:seq_len]
            padded_mask[i, :seq_len] = 1
            
        return padded_ids, padded_mask
    
    async def train(self, input_data, n_generates=4, beta=0.01, clip_param=0.2):
        self.policy_model.train()
        self.ref_model.eval()

        policy_model_result_all = []
        ref_model_result_all = []

        async def generate_single(i):
            policy_model_result = await generate_text_with_probs(
                model=self.policy_model,
                tokenizer=self.tokenizer,
                inputs=input_data,
                device=self.policy_model_device
            )
            ref_model_result = await generate_text_with_probs(
                model=self.ref_model,
                tokenizer=self.tokenizer,
                inputs=input_data,
                device=self.ref_model_device
            )
            policy_model_result_all.append(policy_model_result)
            ref_model_result_all.append(ref_model_result)

        tasks = [generate_single(i) for i in range(n_generates)]
        await asyncio.gather(*tasks)

        # Stack results
        policy_log_probs = torch.cat([result['log_probs'] for result in policy_model_result_all]).to('cpu')  # [n_generates, seq_len, vocab]
        ref_log_probs = torch.cat([result['log_probs'] for result in ref_model_result_all]).to('cpu')  # [n_generates, seq_len, vocab]
        masks = torch.cat([result['completion_mask'] for result in policy_model_result_all]).to('cpu')  # [n_generates, seq_len]
        
        # 统一填充
        generated_ids, _ = self.pad_sequences(policy_model_result_all, 512)
        ref_generated_ids, _ = self.pad_sequences(ref_model_result_all, 512)
        
        print(f"Shapes - policy_log_probs: {policy_log_probs.shape}, "
              f"ref_log_probs: {ref_log_probs.shape}, "
              f"masks: {masks.shape}",
              f"generated_ids: {generated_ids.shape}")

        # Calculate rewards
        rewards = []
        for result in policy_model_result_all:
            reward = self.reward_fn(result['texts'][0])
            rewards.append(reward)
            print("Generated text:", result['texts'][0])
            print("Reward:", reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)  # [n_generates]
        
        # 修改优势值计算，增加数值稳定性
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        print("Rewards stats - mean:", mean_reward.item(), "std:", std_reward.item())
        
        if std_reward < 1e-6:
            print("Warning: rewards have very small variance")
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        
        advantages = advantages.view(-1, 1, 1)  # [n_generates, 1, 1] for broadcasting
        print("Advantages:", advantages.squeeze().tolist())
        
        # Get log probs of generated tokens
        policy_action_log_probs = policy_log_probs.gather(
            -1, 
            generated_ids.unsqueeze(-1).to('cpu')
        ).squeeze(-1)  # [n_generates, seq_len]
        
        ref_action_log_probs = ref_log_probs.gather(
            -1,
            generated_ids.unsqueeze(-1).to('cpu')
        ).squeeze(-1)  # [n_generates, seq_len]
        
        # 计算 ratio 时增加数值稳定性
        log_ratio = policy_action_log_probs - ref_action_log_probs
        # 限制 log_ratio 的范围，防止 exp 后的值过大
        log_ratio = torch.clamp(log_ratio, -20, 20)
        ratio = torch.exp(log_ratio)
        
        # Clip the ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        
        # 打印 ratio 的统计信息
        print("Ratio stats - mean:", ratio.mean().item(), "std:", ratio.std().item(),
              "min:", ratio.min().item(), "max:", ratio.max().item())
        
        # Calculate KL divergence with numerical stability
        kl_div = torch.clamp(policy_action_log_probs - ref_action_log_probs, -100, 100)
        print("KL div stats - mean:", kl_div.mean().item(), "std:", kl_div.std().item())
        
        # Calculate policy loss components
        policy_loss1 = ratio * advantages
        policy_loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2)
        
        # Add KL penalty and apply mask
        total_loss = (policy_loss + beta * kl_div) * masks
        
        # 检查 total_loss 是否包含 nan
        if torch.isnan(total_loss).any():
            print("Warning: NaN detected in total_loss!")
            print("Policy loss stats:", policy_loss.mean().item(), policy_loss.std().item())
            print("KL div contribution:", (beta * kl_div).mean().item())
            return None
        
        # Normalize by the number of non-masked tokens
        mask_sum = masks.sum() + 1e-8
        loss = total_loss.sum() / mask_sum
        
        print("Final loss:", loss.item())
        return loss