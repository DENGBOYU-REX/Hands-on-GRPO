import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def policy_model(model_path:str, lora:bool=True, lora_rank = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj"], device = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if lora:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        # 冻结基础模型权重
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    print ('Policy model loaded')
    print('policy model device:', model.device)
    return model, tokenizer