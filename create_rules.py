import pandas as pd

def create_rules_excel():
    """创建规则Excel文件"""
    rules = [
        {
            'pattern': 'Battery level: (\\d+\\.\\d+)%',
            'name': '电池电量',
            'description': '匹配电池电量信息'
        },
        {
            'pattern': 'User (user\d+) logged in',
            'name': '登陆日志',
            'description': '匹配用户登录信息'
        },
        {
            'pattern': 'User (\\w+) (查看商品|添加购物车|提交订单|支付订单|查看订单|评价商品|搜索商品|收藏商品|分享商品)',
            'name': '用户操作',
            'description': '匹配用户操作行为'
        },
        {
            'pattern': 'WARN.*',
            'name': '系统警告',
            'description': '匹配系统警告信息'
        },
        {
            'pattern': 'ERROR.*',
            'name': '系统错误',
            'description': '匹配系统错误信息'
        }
    ]
    
    df = pd.DataFrame(rules)
    df.to_excel('rules.xlsx', index=False)
    print("规则文件已创建：rules.xlsx")

if __name__ == "__main__":
    create_rules_excel() 