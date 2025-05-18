import random
from datetime import datetime, timedelta
import os

def generate_battery_log():
    """生成电池电量日志"""
    start_time = datetime.now() - timedelta(days=1)
    current_battery = 100
    battery_log = []
    
    for i in range(96):  # 每15分钟一条记录，24小时共96条
        current_time = start_time + timedelta(minutes=i*15)
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 模拟电池消耗
        if current_battery > 0:
            # 随机消耗0.1-0.5的电量
            drain = random.uniform(0.1, 0.5)
            current_battery = max(0, current_battery - drain)
            
            # 随机充电事件
            if random.random() < 0.1 and current_battery < 50:  # 10%概率充电
                current_battery = min(100, current_battery + random.uniform(10, 30))
            
            battery_log.append(f"{time_str} INFO Battery level: {current_battery:.1f}%")
    
    return battery_log

def generate_user_behavior_log():
    """生成用户操作行为日志"""
    start_time = datetime.now() - timedelta(days=1)
    behaviors = [
        "用户登录",
        "查看商品",
        "添加购物车",
        "提交订单",
        "支付订单",
        "查看订单",
        "评价商品",
        "搜索商品",
        "收藏商品",
        "分享商品"
    ]
    
    user_log = []
    current_time = start_time
    
    # 生成24小时内的用户行为
    while current_time < datetime.now():
        # 随机决定是否生成行为
        if random.random() < 0.3:  # 30%概率生成行为
            time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            behavior = random.choice(behaviors)
            user_id = f"user{random.randint(1, 5)}"
            
            if behavior == "用户登录":
                user_log.append(f"{time_str} INFO User {user_id} logged in")
            else:
                user_log.append(f"{time_str} INFO User {user_id} {behavior}")
        
        # 随机增加时间间隔（1-30分钟）
        current_time += timedelta(minutes=random.randint(1, 30))
    
    return user_log

def generate_system_log():
    """生成系统日志"""
    start_time = datetime.now() - timedelta(days=1)
    current_time = start_time
    system_log = []
    
    while current_time < datetime.now():
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 随机生成系统事件
        event_type = random.choices(
            ['INFO', 'WARN', 'ERROR'],
            weights=[0.7, 0.2, 0.1]
        )[0]
        
        if event_type == 'INFO':
            system_log.append(f"{time_str} INFO System running normally")
        elif event_type == 'WARN':
            system_log.append(f"{time_str} WARN High memory usage detected")
        else:
            system_log.append(f"{time_str} ERROR Database connection failed")
        
        # 随机增加时间间隔（5-60分钟）
        current_time += timedelta(minutes=random.randint(5, 60))
    
    return system_log

def create_test_logs():
    """创建测试日志文件"""
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 生成并保存电池日志
    battery_log = generate_battery_log()
    with open('logs/battery.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(battery_log))
    
    # 生成并保存用户行为日志
    user_log = generate_user_behavior_log()
    with open('logs/user_behavior.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(user_log))
    
    # 生成并保存系统日志
    system_log = generate_system_log()
    with open('logs/system.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(system_log))
    
    print("测试日志文件已创建：")
    print("- logs/battery.log")
    print("- logs/user_behavior.log")
    print("- logs/system.log")

if __name__ == "__main__":
    create_test_logs() 