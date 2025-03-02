import os
from typing import List
from openai import OpenAI

# 设置 API 配置
client = OpenAI(
    api_key="dummy",  # 可以是任意值
    base_url="http://localhost:8000/v1"  # 假设qwen模型在本地部署
)

def classify_aspects(review: str, aspect_list: List[str]) -> List[int]:
    """
    使用qwen模型对评论进行多标签分类
    
    Args:
        review: 评论文本
        aspect_list: 需要判断的观点列表
    
    Returns:
        labels: 对应观点列表的二值标签 [1,0,1,0,...]，1表示包含该观点，0表示不包含
    """
    # 构建系统提示词
    system_prompt = """你是一个专业的文本分类助手。你需要判断给定评论中是否包含特定观点。
请仔细分析评论内容，对每个观点进行判断，返回二值标签列表。
1表示评论包含该观点或相关内容，0表示不包含。
只返回标签列表，不要有其他内容。"""
    
    # 构建用户提示词
    aspects_str = "\n".join([f"{i+1}. {aspect}" for i, aspect in enumerate(aspect_list)])
    user_prompt = f"""评论：{review}

需要判断的观点：
{aspects_str}

请返回标签列表，格式示例：[1,0,1,0,...]"""

    try:
        # 调用qwen模型
        response = client.chat.completions.create(
            model="/root/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct",  # 使用qwen模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # 使用确定性输出
            max_tokens=50  # 由于只需要返回标签列表，tokens可以设置较小
        )
        
        # 解析返回的标签列表字符串
        labels_str = response.choices[0].message.content.strip()
        labels = eval(labels_str)  # 将字符串转换为列表
        
        # 验证标签列表
        if not isinstance(labels, list) or len(labels) != len(aspect_list):
            raise ValueError("模型返回的标签列表格式不正确")
        if not all(isinstance(x, int) and x in [0, 1] for x in labels):
            raise ValueError("标签值必须为0或1")
            
        return labels
        
    except Exception as e:
        print(f"分类过程出现错误: {str(e)}")
        # 发生错误时返回全0标签
        return [0] * len(aspect_list)

# 使用示例
if __name__ == "__main__":
    review = "这款耳机音质非常棒，低音浑厚，高音清晰。续航也很给力，充一次能用好几天。就是价格稍微贵了点。"
    aspects = ["音质", "续航", "舒适度", "性价比", "做工"]
    
    labels = classify_aspects(review, aspects)
    print(f"评论：{review}")
    print(f"观点列表：{aspects}")
    print(f"分类结果：{labels}")
