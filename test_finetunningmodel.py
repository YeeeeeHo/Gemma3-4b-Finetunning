from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ 병합된 모델 로드
model = AutoModelForCausalLM.from_pretrained("./merged_model", device_map="auto")

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", token="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")

# ✅ 추론 프롬프트
prompt = ""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ✅ 추론 with 반복 억제 옵션
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True  # 꼭 있어야 다양화됨
    )

# ✅ 출력 결과
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
