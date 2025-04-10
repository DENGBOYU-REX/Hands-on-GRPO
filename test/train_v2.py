import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

device = 'cuda'
device_cpu = 'cpu'

def pad_sequences(old_logprobs, new_logprobs):
    """
    填充 old_logprobs 和 new_logprobs 的第二个维度，使其长度相同。
    较短的序列会在第二个维度上补零。
    
    参数:
    - old_logprobs: torch.Tensor, 旧策略的概率分布
    - new_logprobs: torch.Tensor, 新策略的概率分布
    
    返回:
    - padded_old_logprobs: torch.Tensor, 填充后的旧策略概率分布
    - padded_new_logprobs: torch.Tensor, 填充后的新策略概率分布
    """
    # 获取两个张量在第二个维度上的长度
    old_len = old_logprobs.size(1)
    new_len = new_logprobs.size(1)
    
    # 计算需要填充的长度
    max_len = max(old_len, new_len)
    pad_old = max_len - old_len
    pad_new = max_len - new_len
    
    # 如果需要填充，则执行填充操作
    if pad_old > 0:
        padding = (0, 0, 0, pad_old)  # (padding_left, padding_right, padding_top, padding_bottom)
        padded_old_logprobs = torch.nn.functional.pad(old_logprobs, padding, "constant", 0)
    else:
        padded_old_logprobs = old_logprobs
    
    if pad_new > 0:
        padding = (0, 0, 0, pad_new)
        padded_new_logprobs = torch.nn.functional.pad(new_logprobs, padding, "constant", 0)
    else:
        padded_new_logprobs = new_logprobs
    
    return padded_old_logprobs, padded_new_logprobs

class GRPOTrainer:
    def __init__(self, model_name, reward_fn, num_generations=2, beta=0.1, mu=2, lora=True):
        # 初始化模型和tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto').bfloat16()
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto').bfloat16()
        self.epsilon_low = 0.2
        self.epsilon_high = 0.2
        if lora:
            # LoRA配置
            lora_config = LoraConfig(
                r=4, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config).bfloat16().to(device)

            # 冻结基础模型权重
            for name, param in self.model.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='auto')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # 训练参数
        self.reward_fn = reward_fn  # 自定义规则奖励函数
        self.num_generations = num_generations  # 每组生成数量G
        self.beta = beta  # KL散度系数
        self.mu = mu  # GRPO内部迭代次数
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

    def generate_responses(self, questions):
        """为每个问题生成G个响应"""
        outputs = []
        for q in questions:
            inputs = self.tokenizer(q, return_tensors="pt", padding=True).to(device)
            
            # 使用多样化的采样策略生成多个响应
            generated = self.model.generate(
                inputs.input_ids,
                max_length=2048,
                num_return_sequences=self.num_generations,
                do_sample=True,
                top_p=0.7,
                temperature=0.5,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            
            # 解析生成结果
            sequences = generated.sequences
            logits = torch.stack(generated.scores, dim=1)
            outputs.append((sequences.to(device_cpu), logits.to(device_cpu)))
            del sequences, logits, generated, inputs
            torch.cuda.empty_cache()

        return outputs

    def compute_advantages(self, rewards):
        """组相对优势估计（简化实现）"""
        print('rewards:\n', rewards, '\n')
        advantages = []
        for group_rewards in rewards:
            mean_reward = torch.mean(group_rewards)
            std_reward = torch.std(group_rewards) + 1e-8
            advantages_group = (group_rewards - mean_reward) / std_reward
            advantages.extend(advantages_group)
        return torch.stack(advantages)
    
    def compute_kl_pre_token(self, logprobs, ref_logprobs):
        """计算KL散度"""
        per_token_kl = (
            torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        )
        coef_1 = torch.exp(logprobs - ref_logprobs)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        return per_token_kl, coef_1, coef_2

    def grpo_loss(self, logprobs, old_logprobs, advantages, completion_mask):
        """GRPO损失函数（含KL惩罚）"""
        logprobs = torch.softmax(logprobs, dim=-1)
        old_logprobs = torch.softmax(old_logprobs, dim=-1)

        ratio = torch.exp(logprobs.to(device_cpu) - old_logprobs.detach().to(device_cpu))
        ratio = torch.clamp(ratio, 0.1, 10)  # 防止极端值

        kl_penalty, coef_1, coef_2 = self.compute_kl_pre_token(logprobs.to(device_cpu), old_logprobs.to(device_cpu))

        per_token_loss1 = coef_1 * advantages.unsqueeze(1).to(device_cpu)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1).to(device_cpu)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        per_token_loss = per_token_loss + self.beta * kl_penalty

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        if torch.isnan(kl_penalty).any():
            print("KL计算出现NaN！")
            kl_penalty = torch.zeros_like(kl_penalty)

        policy_loss = -torch.mean(ratio.to(device_cpu) * advantages.unsqueeze(-1).to(device_cpu))
        if torch.isnan(policy_loss).any():
            print("policy_loss计算出现NaN！")
            policy_loss = torch.zeros_like(policy_loss)
        total_loss = policy_loss + self.beta * kl_penalty.mean()

        return total_loss

    def prepare_inputs_for_training(self, prompts, answers):
        """Prepare inputs for training including completion masks and reference log probs"""
        all_input_ids = []
        all_attention_masks = []
        all_completion_masks = []
        all_ref_per_token_logps = []

        for prompt, answer in zip(prompts, answers):
            full_text = f"{prompt}##Think##{answer}"
            inputs = self.tokenizer(full_text, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            
            # Determine the start of the completion part
            prompt_ids = self.tokenizer.encode(f"{prompt}##Think##")
            completion_start_idx = len(prompt_ids)
            completion_mask = torch.zeros_like(input_ids)
            completion_mask[completion_start_idx:] = 1
            
            # Get reference model log probabilities
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids.unsqueeze(0).to(device), attention_mask=attention_mask.unsqueeze(0).to(device))
                # ref_logits = ref_outputs.logits.squeeze(0)[:, :-1, :]
                ref_logits = ref_outputs.logits.squeeze(0)
                target_ids = input_ids[1:].unsqueeze(0)
                ref_log_probs = torch.gather(torch.log_softmax(ref_logits.to(device_cpu), dim=-1), -1, target_ids.unsqueeze(-1).to(device_cpu)).squeeze(-1)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_completion_masks.append(completion_mask)
            all_ref_per_token_logps.append(ref_log_probs.cpu())

        # Pad sequences
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = [torch.cat([seq, torch.full((max_len - len(seq),), self.tokenizer.pad_token_id)]) for seq in all_input_ids]
        padded_attention_masks = [torch.cat([mask, torch.zeros(max_len - len(mask))]) for mask in all_attention_masks]
        padded_completion_masks = [torch.cat([mask, torch.zeros(max_len - len(mask))]) for mask in all_completion_masks]
        padded_ref_per_token_logps = [torch.cat([log_probs, torch.zeros(max_len - len(log_probs))]) for log_probs in all_ref_per_token_logps]

        inputs = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'completion_mask': torch.stack(padded_completion_masks),
            'ref_per_token_logps': torch.stack(padded_ref_per_token_logps)
        }
        return inputs

    def train_step(self, questions):
        # 生成阶段
        self.model.eval()
        generated_data = self.generate_responses(questions)
        
        # 准备答案
        answers = []
        for (sequences, _) in generated_data:
            decoded_texts = [self.tokenizer.decode(seq, skip_special_tokens=True) 
                            for seq in sequences]
            answers.append(decoded_texts)
        
        # 准备训练输入
        all_prompts = questions * self.num_generations
        all_answers = [ans for ans_group in answers for ans in ans_group]
        inputs = self.prepare_inputs_for_training(all_prompts, all_answers)
        
        # 计算奖励
        rewards = []
        for ans_group in answers:
            group_rewards = torch.tensor(
                [self.reward_fn(text) for text in ans_group],
                dtype=torch.float16
            )
            rewards.append(group_rewards)
        
        # 计算优势
        advantages = self.compute_advantages(rewards)
        inputs['advantages'] = advantages
        
        # 获取旧策略的概率分布
        with torch.no_grad():
            old_logits = self.model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device)).logits
            old_logprobs = torch.log_softmax(old_logits, dim=-1)
        inputs['old_per_token_logps'] = old_logprobs.cpu().detach()
        
        # 策略优化阶段
        self.model.train()
        for _ in range(self.mu):
            losses = []
            for i in range(0, len(inputs['input_ids']), self.num_generations):
                batch_inputs = {key: value[i:i+self.num_generations] for key, value in inputs.items()}
                
                loss = self.grpo_loss(
                    new_logprobs=batch_inputs['old_per_token_logps'][:, :-1, :].to(device),
                    old_logprobs=batch_inputs['ref_per_token_logps'][:, 1:, :].to(device),
                    advantages=batch_inputs['advantages'][i:i+self.num_generations].unsqueeze(-1).to(device),
                    completion_mask=batch_inputs['completion_mask'][:, 1:].to(device)
                )
                print("loss: ", loss.item())
                losses.append(loss.to(device))
                del loss
                torch.cuda.empty_cache()
            
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            del losses, total_loss
            torch.cuda.empty_cache()

        # # 更新参考模型每 mu 次内部迭代更新一次
        # self.ref_model.load_state_dict(self.model.state_dict())

# 使用示例
if __name__ == "__main__":
    def dummy_reward_fn(text):
        return 1.0 if "##Think##" in text and '##Answer##' in text else 0.0

    trainer = GRPOTrainer(
        model_name="/data1/common_models/Qwen/Qwen2.5-3B-Instruct/",
        reward_fn=dummy_reward_fn,
        num_generations=2,
        beta=0.1,
        mu=2
    )
    
    task_prompts = [
        "请解释相对论的核心思想, output your thinking process after ##Think##, then output your answer after ##Answer##",
        "如何证明勾股定理？ output your thinking process after ##Think##, then output your answer after ##Answer##",
        "描述量子纠缠的基本原理： output your thinking process after ##Think##, then output your answer after ##Answer##"
    ]

    for iteration in tqdm(range(50)):
        print(f"Iteration {iteration+1}")
        trainer.train_step(task_prompts)



