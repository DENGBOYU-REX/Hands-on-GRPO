from train_edit import Trainer
import asyncio

trainer = Trainer('/data1/common_models/Qwen/Qwen2.5-3B-Instruct/', '/data1/common_models/Qwen/Qwen2.5-7B-Instruct/')
# trainer.train(input_data=["Translate English to French: Hello, how are you today?"])
# 异步运行训练
n_generates = 2  # 生成次数
loss = asyncio.run(trainer.train(input_data=["Translate English to French: Hello, how are you today?, output the result after ##Result##, output reason after ##Reason##"], n_generates=n_generates))

# print (policy_results, ref_results)
# loss = await trainer.train(
#     input_data=["your input text"],
#     n_generates=4,
#     beta=0.01,
#     clip_param=0.2
# )