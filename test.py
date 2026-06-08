from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
lwm = "/home/admin/workspace/aop_lab/app_data/.cache/models--LargeWorldModel--LWM-Text-Chat-1M/snapshots/0598c443b02aeb1a1f9f6788e9af85ea762a452d"
qwen_8B = "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
llama2_7b = "/home/admin/workspace/aop_lab/app_data/.cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
model_id = lwm


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).cuda()

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)
input_ids = tokenizer.encode("China is")
input_ids = torch.tensor([input_ids],dtype = torch.int64).cuda()
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
