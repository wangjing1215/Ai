import pandas as pd
import random
from datetime import datetime, timedelta
import os

def create_rules_excel():
    """创建规则Excel文件"""
    rules = [
        {
            'pattern': 'ERROR.*',
            'name': '错误日志',
            'description': '匹配错误信息'
        },
        {
            'pattern': 'WARN.*',
            'name': '警告日志',
            'description': '匹配警告信息'
        },
        {
            'pattern': 'INFO.*',
            'name': '信息日志',
            'description': '匹配普通信息'
        },
        {
            'pattern': 'Exception: (.*)',
            'name': '异常日志',
            'description': '匹配异常信息'
        },
        {
            'pattern': 'User (.*) logged in',
            'name': '用户登录',
            'description': '匹配用户登录信息'
        }
    ]
    
    df = pd.DataFrame(rules)
    df.to_excel('rules.xlsx', index=False)
    print("规则文件已创建：rules.xlsx")

def create_test_logs():
    """创建测试日志文件"""
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成app.log
    with open('logs/app.log', 'w', encoding='utf-8') as f:
        start_time = datetime.now() - timedelta(days=1)
        for i in range(100):
            current_time = start_time + timedelta(minutes=i*15)
            time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 随机生成不同类型的日志
            log_type = random.choice(['INFO', 'WARN', 'ERROR'])
            if log_type == 'INFO':
                if random.random() < 0.3:  # 30%概率生成登录日志
                    user = f"user{random.randint(1, 5)}"
                    f.write(f"{time_str} INFO User {user} logged in\n")
                else:
                    f.write(f"{time_str} INFO System running normally\n")
            elif log_type == 'WARN':
                f.write(f"{time_str} WARN High memory usage detected\n")
            else:
                f.write(f"{time_str} ERROR Database connection failed\n")
    
    # 生成error.log
    with open('logs/error.log', 'w', encoding='utf-8') as f:
        start_time = datetime.now() - timedelta(days=1)
        for i in range(50):
            current_time = start_time + timedelta(minutes=i*30)
            time_str = current_time.strftime('[%Y/%m/%d %H:%M:%S]')
            
            # 生成错误日志
            error_type = random.choice(['Exception', 'Error', 'Critical'])
            if error_type == 'Exception':
                f.write(f"{time_str} Exception: Connection timeout\n")
            elif error_type == 'Error':
                f.write(f"{time_str} Error: Invalid input data\n")
            else:
                f.write(f"{time_str} Critical: System crash detected\n")
    
    print("测试日志文件已创建：")
    print("- logs/app.log")
    print("- logs/error.log")

def main():
    create_rules_excel()
    create_test_logs()

if __name__ == "__main__":
    main() 