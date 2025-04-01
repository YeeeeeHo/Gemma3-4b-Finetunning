from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-3-4b-pt"
hf_token = "hf_..."  # 🔐 Hugging Face 토큰

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_token,
    trust_remote_code=True
)

# ✅ 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ✅ 프롬프트 (instruction 아님)
prompt = "서울에서 맛집 추천해줘"

# ✅ 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ✅ 추론
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

# ✅ 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
