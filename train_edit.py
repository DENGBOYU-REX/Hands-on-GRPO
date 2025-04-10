from utils.policy_model import policy_model
from utils.generate_probs_mask import generate_text_with_probs
from utils.ref_model import ref_model
from utils.reward_func import dummy_reward_fn
from loss.kl_loss import compute_kl_divergence as per_token_kl
import torch
import asyncio

class Trainer:
    def __init__(self, model_path, ref_model_path, ref_model_device='cuda:0', policy_model_device='cuda:1', reward_fn=dummy_reward_fn):
        self.policy_model, self.tokenizer = policy_model(model_path, device=policy_model_device)
        self.ref_model, _ = ref_model(ref_model_path, device=ref_model_device)
        self.reward_fn = reward_fn
        self.ref_model_device = ref_model_device
        self.policy_model_device = policy_model_device
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=1e-5)
        
    async def train(self, input_data, n_generates=4, beta=0.01, clip_param=0.2):
        self.policy_model.train()
        self.ref_model.eval()
        
        # 1. 生成样本
        policy_outputs = await self.generate_samples(input_data, n_generates)
        
        # 2. 计算奖励和优势
        rewards = self.compute_rewards(policy_outputs)
        advantages = self.compute_advantages(rewards)
        
        # 3. 计算策略梯度
        loss = self.compute_policy_gradient(policy_outputs, advantages, beta, clip_param)
        
        # 4. 更新模型
        if not torch.isnan(loss).any():
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            
        return loss.item() if not torch.isnan(loss).any() else None
        
    async def generate_samples(self, input_data, n_generates):
        """生成样本并获取概率分布"""
        outputs = []
        for _ in range(n_generates):
            output = await generate_text_with_probs(
                model=self.policy_model,
                tokenizer=self.tokenizer,
                inputs=input_data,
                device=self.policy_model_device
            )
            outputs.append(output)
        return outputs
        
    def compute_rewards(self, outputs):
        """计算每个样本的奖励"""
        rewards = []
        for output in outputs:
            reward = self.reward_fn(output['texts'][0])
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
        
    def compute_advantages(self, rewards):
        """计算优势值"""
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward < 1e-6:
            return torch.zeros_like(rewards)
        return (rewards - mean_reward) / (std_reward + 1e-8)
        
    def pad_sequences(self, sequences, max_len):
        """填充序列到相同长度"""
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = torch.zeros(max_len - len(seq), seq.size(-1))
                padded.append(torch.cat([seq, padding]))
            else:
                padded.append(seq[:max_len])
        return torch.stack(padded)

    def get_ref_log_probs(self, outputs):
        """获取参考模型的对数概率，并处理不同长度的序列"""
        with torch.no_grad():
            ref_log_probs = []
            max_len = max(output['generated_ids'].size(1) for output in outputs)
            
            for output in outputs:
                # 获取生成的ID序列并确保填充到最大长度
                input_ids = output['generated_ids'].to(self.ref_model_device)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                
                # 获取参考模型的输出
                ref_output = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 计算log概率并处理填充
                logits = ref_output.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # 确保所有序列长度一致
                if log_probs.size(1) < max_len:
                    padding = torch.zeros(
                        log_probs.size(0),
                        max_len - log_probs.size(1),
                        log_probs.size(2),
                        device=log_probs.device
                    )
                    log_probs = torch.cat([log_probs, padding], dim=1)
                else:
                    log_probs = log_probs[:, :max_len]
                
                ref_log_probs.append(log_probs)
            
            return torch.cat(ref_log_probs)

    def compute_policy_gradient(self, outputs, advantages, beta, clip_param):
        """计算策略梯度损失"""
        # 获取最大序列长度
        max_len = max(output['log_probs'].size(1) for output in outputs)
        
        # 获取并填充策略模型的log概率
        policy_log_probs = []
        masks = []
        for output in outputs:
            log_probs = output['log_probs']
            mask = output['completion_mask']
            
            # 填充到最大长度
            if log_probs.size(1) < max_len:
                padding = torch.zeros(
                    log_probs.size(0),
                    max_len - log_probs.size(1),
                    log_probs.size(2),
                    device=log_probs.device
                )
                log_probs = torch.cat([log_probs, padding], dim=1)
                
                mask_padding = torch.zeros(
                    mask.size(0),
                    max_len - mask.size(1),
                    device=mask.device
                )
                mask = torch.cat([mask, mask_padding], dim=1)
            
            policy_log_probs.append(log_probs)
            masks.append(mask)
        
        policy_log_probs = torch.cat(policy_log_probs)
        masks = torch.cat(masks)
        
        # 获取参考模型的log概率
        ref_log_probs = self.get_ref_log_probs(outputs)
        
        # 确保维度匹配
        assert policy_log_probs.size() == ref_log_probs.size(), \
            f"Shape mismatch: policy {policy_log_probs.size()} vs ref {ref_log_probs.size()}"
        
        # 计算比率并裁剪
        log_ratio = torch.clamp(policy_log_probs - ref_log_probs, -20, 20)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        
        # 计算策略损失
        policy_loss1 = ratio * advantages.view(-1, 1, 1)
        policy_loss2 = clipped_ratio * advantages.view(-1, 1, 1)
        policy_loss = -torch.min(policy_loss1, policy_loss2)
        
        # 计算KL散度
        kl_div = self.compute_kl_divergence(policy_log_probs, ref_log_probs)
        
        # 组合损失并应用掩码
        total_loss = (policy_loss + beta * kl_div) * masks.unsqueeze(-1)
        
        # 归一化
        return total_loss.sum() / (masks.sum() + 1e-8)