import os
import json
import numpy as np
from typing import List, Tuple
from openai import OpenAI
from tqdm import tqdm

# 设置 API 配置
client = OpenAI(
    api_key="dummy",  # 可以是任意值
    base_url="http://localhost:8000/v1"
)

# 定义8个方面的类别
ASPECT_CATEGORIES = [
    '音质', '续航', '舒适度', '通话质量',
    '蓝牙连接', '性价比', '做工质量', '其他'
]

import random

def get_random_prompt() -> Tuple[str, str]:
    """随机生成系统提示词和用户提示词"""
    # 多个场景变体
    scenarios = [
        "通勤路上", "健身房锻炼", "咖啡厅办公", "地铁上", "跑步时",
        "居家办公", "图书馆学习", "户外运动", "视频会议", "游戏时"
    ]
    
    # 多个价格区间变体
    price_ranges = [
        "百元以下", "200-500元", "500-1000元", "1000-2000元", "2000元以上"
    ]
    
    # 多个品牌对比变体
    brands = [
        "AirPods", "华为FreeBuds", "小米耳机", "SONY WF", "Beats",
        "三星Galaxy Buds", "OPPO Enco", "1MORE", "QCY", "FIIL"
    ]
    
    # 随机选择要关注的方面数量和具体方面
    aspects = random.sample(ASPECT_CATEGORIES[:-1], random.randint(2, 4))
    
    system_prompt = f"""你是一个专业的蓝牙耳机评论生成器。你需要生成真实、自然、详细的蓝牙耳机评论。
    评论应该：
    1. 重点关注这些方面：{', '.join(aspects)}
    2. 场景设定在：{random.choice(scenarios)}
    3. 价格区间在：{random.choice(price_ranges)}
    4. 对比参考：{random.choice(brands)}
    5. 使用口语化表达，避免过于专业的术语
    6. 保持观点的独特性，避免套话
    7. 评论长度控制在50-80字之间"""
    
    user_prompt = f"""请生成一条关于蓝牙耳机的用户评论，要求如下：
    1. 必须详细描述这些方面：{', '.join(aspects)}
    2. 评论长度控制在50-80字之间
    3. 使用具体的数据支撑，如：
       - 使用时长（小时）
       - 充电时间（分钟）
       - 延迟数据（毫秒）
       - 通话距离（米）
       - 价格（元）等
    4. 加入个人使用场景和体验
    5. 描述要具体，不要泛泛而谈
    6. 只返回评论内容，不要包含其他内容"""
    
    return system_prompt, user_prompt

def is_valid_review(review: str) -> bool:
    """验证评论是否有效"""
    if not review or not isinstance(review, str):
        return False
    
    # 检查长度
    if len(review) < 50 or len(review) > 80:
        return False
    
    # 检查是否包含数字（具体数据）
    if not any(c.isdigit() for c in review):
        return False
    
    # 检查是否包含常见无效内容
    invalid_patterns = [
        "这是一条评论",
        "好评",
        "差评",
        "以下是",
        "评论内容：",
        "总的来说",
    ]
    if any(pattern in review for pattern in invalid_patterns):
        return False
    
    return True

def generate_review(max_retries: int = 3) -> Tuple[str, List[int]]:
    """生成一条蓝牙耳机评论及其对应的标签"""
    for attempt in range(max_retries):
        try:
            # 获取随机生成的提示词
            system_prompt, user_prompt = get_random_prompt()
            
            # 生成评论
            print("生成评论中: ", end="", flush=True)
            review_stream = client.chat.completions.create(
                model="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=random.uniform(0.8, 1.0),  # 随机温度
                max_tokens=300,   # 增加长度上限
                stream=True,
                presence_penalty=random.uniform(0.6, 0.8),  # 随机presentce_penalty
                frequency_penalty=random.uniform(0.3, 0.5),  # 随机frequency_penalty
                top_p=random.uniform(0.9, 1.0)  # 随机top_p
            )
            
            review = ""
            for chunk in review_stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    review += content
                    print(content, end="", flush=True)
            print("\n")
            
            review = review.strip()
            
            # 为评论生成标签
            label_prompt = f"""对于以下蓝牙耳机评论：
            "{review}"
            
            请判断该评论涉及了以下哪些方面（用0和1表示）：
            音质、续航、舒适度、通话质量、蓝牙连接、性价比、做工质量、其他
            只返回一行数字，用逗号分隔，例如：1,0,0,1,0,0,0,0"""
            
            print("生成标签中: ", end="", flush=True)
            label_stream = client.chat.completions.create(
                model="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": label_prompt}
                ],
                temperature=0.1,
                max_tokens=50,
                stream=True
            )
            
            label_text = ""
            for chunk in label_stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    label_text += content
                    print(content, end="", flush=True)
            print("\n")
            
            labels = [int(x) for x in label_text.strip().split(',')]
            return review, labels
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                return None, None
            continue
    return None, None  # 如果所有尝试都失败

def save_review(review: str, label: List[int], output_dir: str, current_count: int):
    """保存单条评论数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到临时文件
    temp_file = os.path.join(output_dir, "reviews_temp.jsonl")
    with open(temp_file, "a", encoding="utf-8") as f:
        data = {
            "id": current_count,
            "text": review,
            "label": label
        }
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def generate_dataset(num_samples: int = 10, output_dir: str = "/root/code/bert/data/train") -> None:
    """生成指定数量的评论数据集并实时保存"""
    # 读取已生成的评论
    temp_file = os.path.join(output_dir, "reviews_temp.jsonl")
    existing_reviews = []
    current_count = 0
    
    if os.path.exists(temp_file):
        with open(temp_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                existing_reviews.append(data["text"])
                current_count = max(current_count, data["id"] + 1)
    
    total_needed = num_samples - current_count
    if total_needed <= 0:
        print(f"已经生成了足够的数据: {current_count} 条")
        return
    
    with tqdm(total=total_needed, desc=f"继续生成数据(当前{current_count}条)") as pbar:
        while current_count < num_samples:
            review, label = generate_review()
            if review and label and len(label) == 8 and is_valid_review(review):
                # 检查是否与已有评论过于相似
                is_duplicate = False
                for existing_review in existing_reviews[-10:]:
                    if len(set(review) & set(existing_review)) / len(set(review)) > 0.8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    save_review(review, label, output_dir, current_count)
                    existing_reviews.append(review)
                    current_count += 1
                    pbar.update(1)
    
    # 生成完成后，将临时文件转换为最终文件
    final_file = os.path.join(output_dir, "train.json")
    final_data = {
        "texts": [],
        "labels": []
    }
    
    with open(temp_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            final_data["texts"].append(data["text"])
            final_data["labels"].append(data["label"])
    
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据生成完成，共 {len(final_data['texts'])} 条评论已保存到 {final_file}")

def save_dataset(texts: List[str], labels: List[List[int]], output_dir: str):
    """保存数据集到文件"""
    data = {
        "texts": texts,
        "labels": labels,
        "aspect_categories": ASPECT_CATEGORIES
    }
    
    output_file = os.path.join(output_dir, "bluetooth_headphone_reviews.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    # 生成5000条评论数据
    generate_dataset(5000)
