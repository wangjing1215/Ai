tools = {
    "tools": [
        {
            "name": "文生图",
            "description": "一个将文本输入转换为图像生成的工具，可用于创建插图或视觉内容。",
            "parameters": {
                "input_text": "用户输入的描述性文本，用于生成图像。",
                "style": "生成图像的艺术风格选择，例如现代、经典、卡通等。",
                "resolution": "生成图像的分辨率设置。",
                "output_format": "输出的图像格式，支持 JPEG、PNG 等格式。"
            }
        },
        {
            "name": "图生文",
            "description": "将图像转换为文本描述的工具，适用于图像内容分析和说明。",
            "parameters": {
                "input_image": "用户上传的图像文件。",
                "description_length": "生成文本的长度选择，例如简短、中等、详细。",
                "language": "输出文本的语言选择。",
                "output_format": "文本输出格式选择，例如纯文本或富文本。"
            }
        },
        {
            "name": "文生音频",
            "description": "将文本内容转换为音频文件的工具，适用于创建有声读物或演示。",
            "parameters": {
                "input_text": "用户输入的文本内容，用于生成音频。",
                "voice_type": "选择生成音频的声音类型，例如男声、女声或机器人声。",
                "speed": "音频播放速度的调整，例如正常、慢速、快速。",
                "output_format": "输出的音频格式，支持 MP3、WAV 等格式。"
            }
        },
        {
            "name": "音频生文",
            "description": "将音频内容转换为文本的工具，适用于转录会议、讲座等音频材料。",
            "parameters": {
                "input_audio": "用户上传的音频文件，需要转录为文本。",
                "language": "音频内容的语言选择，以提高转录准确性。",
                "transcription_mode": "转录模式选择，例如逐字转录、摘要转录。",
                "output_format": "文本输出格式选择，例如纯文本、富文本或文本文件。"
            }
        }
    ]
}

prompt = """
你是一个工作助手，你有如下工具：
{}
请你根据用户的需求，编排合适的工作流程
如：
1.文生图
2.图生文
使用单个或多个工具来完成需求

你只能输出工作的编排计划，不要输出其他的解释性语句
"""

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)

chat_completion = client.chat.completions.create(
  messages=[
    {
      'role': 'system',
      'content': prompt.format(tools),
    },
    {
      'role': 'user',
      'content': '帮我画一个小女孩',
    }
  ],
  model='qwen2:0.5B',
)
print(chat_completion.choices[0].message.content)