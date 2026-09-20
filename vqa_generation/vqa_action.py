"""
Action VQA Generator
- 입력: val_construction.json / train_construction.json 
- 출력: vqa_action_val.json / vqa_action_train.json 
- action.label이 null인 데이터(안전보조장비)는 제외

Usage:
  python generate_vqa_action.py --input_dir ./json파일 --output_dir ./vqa
"""
import json
import os
import random
import argparse

random.seed(42)

def generate_action_vqa(item):
    """하나의 데이터에서 Action 관련 멀티턴 VQA conversations 생성"""
    act = item.get("action", {})
    label = act.get("label")
    is_danger = act.get("is_danger", False)
    description = act.get("description", "")
    code = act.get("code", "")

    # action.label이 null이면 건너뛰기
    if label is None:
        return None

    conversations = []

    # ── Q1: 위험 여부 판별  ──
    q1_templates = [
        "Is there a dangerous situation in this image?",
        "Does this image show a hazardous condition?",
        "Is the work environment in this image safe?",
    ]
    q1 = random.choice(q1_templates)

    if "safe" in q1.lower():
        # "Is it safe?" 질문이면 답이 반대로
        a1 = "No, it is not safe." if is_danger else "Yes, it is safe."
    else:
        a1 = "Yes." if is_danger else "No, the situation appears safe."

    conversations.append({"from": "human", "value": "<image>\n" + q1})
    conversations.append({"from": "vlm", "value": a1})

    # ── Q2: 사고유형 분류 ──
    q2_templates = [
        "What type of accident risk is present?",
        "What kind of hazard can be identified?",
        "What safety risk does this scene involve?",
    ]
    q2 = random.choice(q2_templates)

    if is_danger:
        a2 = label + "."
    else:
        a2 = "No immediate accident risk. The scene is related to " + label + " but currently appears safe."

    conversations.append({"from": "human", "value": q2})
    conversations.append({"from": "vlm", "value": a2})

    # ── Q3: Safe vs Hazardous 판별 ──
    q3 = "Is this a safe or hazardous work situation?"
    a3 = "Hazardous." if is_danger else "Safe."

    conversations.append({"from": "human", "value": q3})
    conversations.append({"from": "vlm", "value": a3})

    # ── Q4: 상황 서술 ──
    q4_templates = [
        "Describe the hazard shown in this image.",
        "Explain the safety situation in this scene.",
        "What is happening in terms of safety in this image?",
    ]
    q4 = random.choice(q4_templates)
    a4 = description if description else "No detailed description available."

    conversations.append({"from": "human", "value": q4})
    conversations.append({"from": "vlm", "value": a4})

    return {
        "id": item["id"],
        "image": item["image"],
        "conversations": conversations,
    }


def process_file(input_path, output_path):
    """JSON 파일을 읽고 Action VQA를 생성"""
    print(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total items: {len(data)}")

    vqa_data = []
    skipped = 0
    for item in data:
        result = generate_action_vqa(item)
        if result:
            vqa_data.append(result)
        else:
            skipped += 1

    print(f"Generated: {len(vqa_data)} VQA samples")
    print(f"Skipped (null label): {skipped}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vqa_data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Action VQA dataset from unified JSON")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to folder containing val_construction.json & train_construction.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output folder for VQA files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_file(
        os.path.join(args.input_dir, "val_construction.json"),
        os.path.join(args.output_dir, "vqa_action_val.json"),
    )
    process_file(
        os.path.join(args.input_dir, "train_construction.json"),
        os.path.join(args.output_dir, "vqa_action_train.json"),
    )

