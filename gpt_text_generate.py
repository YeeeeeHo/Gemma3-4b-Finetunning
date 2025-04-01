from openai import OpenAI
import json
import re
from tqdm import tqdm
import time
import random

# OpenAI 객체 생성
client = OpenAI(api_key="sk-...")

output_file = "vision_aid_1000.jsonl"
total_rounds = 334
success_count = 0
fail_count = 0

categories = [
    "보행 안전 관련 (예: '지금 길 건너도 될까?', '앞에 장애물 없어?')",
    "주변 환경 파악 (예: '지금 어디쯤이야?', '내 주변에 어떤 게 보여?')",
    "대중교통 이용 (예: '버스가 근처에 있어?', '지하철역 입구가 어디야?')",
    "사물 식별 및 위치 (예: '휴지통 어디 있어?', '내 지팡이 찾아줄래?')",
    "방향 안내 및 길찾기 (예: '오른쪽 길로 가도 돼?', '출구가 어느 쪽이야?')"
]

with open(output_file, "w", encoding="utf-8") as f:
    for i in tqdm(range(total_rounds), desc="Generating"):
        try:
            selected_categories = random.sample(categories, 3)

            dynamic_prompt = f"""
당신은 시각장애인을 돕는 AI 비서입니다.
다음 3가지 질문 유형 각각에 대해 질문(instruction), 시각 인식 정보(input), 적절한 응답(output)을 하나씩 생성해 주세요.

선택된 질문 유형:
- {selected_categories[0]}
- {selected_categories[1]}
- {selected_categories[2]}

형식:
{{"instruction": "...", "input": "...", "output": "..."}}

조건:
- input은 반드시 거리, 방향, 사물 정보를 포함해야 합니다.
- output은 input을 정확히 반영하여 사용자에게 친절하고 정확하게 안내해야 합니다.
- input 내용이 없는 경우 "잠시만 기다려주세요"와 같이 안내합니다.
- 질문은 가능한 한 다양하고 자연스러워야 합니다.
"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": dynamic_prompt}],
                temperature=0.9
            )

            content = response.choices[0].message.content

            # JSON 형태 추출
            matches = re.findall(r'\{.*?\}', content, re.DOTALL)
            count_this_round = 0

            for match in matches:
                try:
                    obj = json.loads(match)
                    if all(k in obj for k in ["instruction", "input", "output"]):
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        count_this_round += 1
                except Exception as e:
                    continue

            success_count += count_this_round
            print(f"✅ {i+1}/{total_rounds} → {count_this_round}개 저장됨 (누적: {success_count})")

            time.sleep(0.7)

        except Exception as e:
            print(f"❌ {i+1}/{total_rounds} 실패 → {e}")
            fail_count += 1
            time.sleep(5)

print(f"\n총 저장된 데이터: {success_count}개 ✅")
print(f"총 실패 요청: {fail_count}개 ❌")
