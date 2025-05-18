import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path
import matplotlib.font_manager as fm

class LogAnalyzer:
    def __init__(self, rules_file, log_dir):
        self.rules_file = rules_file
        self.log_dir = log_dir
        self.rules = None
        self.log_data = []
        self.matches = []
        
    def load_rules(self):
        """从Excel文件加载正则规则"""
        try:
            self.rules = pd.read_excel(self.rules_file)
            # 确保包含is_show和translate列，如果不存在则添加默认值
            if 'is_show' not in self.rules.columns:
                self.rules['is_show'] = 'Y' # 默认显示
            if 'translate' not in self.rules.columns:
                self.rules['translate'] = '%s' # 默认使用匹配值本身
            print(f"成功加载 {len(self.rules)} 条规则")
        except Exception as e:
            print(f"加载规则文件时出错: {str(e)}")
            
    def get_log_files(self):
        """获取日志目录下的所有日志文件"""
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        return log_files
            
    def parse_log(self, log_file):
        """解析单个日志文件"""
        try:
            # 获取日志文件名
            log_name = os.path.basename(log_file)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    for _, rule in self.rules.iterrows():
                        pattern = rule['pattern']
                        rule_name = rule['name']
                        
                        match = re.search(pattern, line)
                        if match:
                            # 尝试从日志中提取时间
                            time_match = re.search(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', line)
                            timestamp = None
                            if time_match:
                                try:
                                    timestamp = datetime.strptime(time_match.group(), '%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    pass
                            
                            # 打印匹配到的日志
                            print(f"\n匹配到日志:")
                            print(f"规则: {rule_name}")
                            print(f"时间: {timestamp}")
                            print(f"内容: {line.strip()}")
                            print(f"匹配组: {match.groups() if match.groups() else '无'}")
                            print("-" * 50)
                            
                            self.log_data.append({
                                'timestamp': timestamp,
                                'rule_name': rule_name,
                                'log_file': log_name,
                                'line': line.strip(),
                                'matches': match.groups() if match.groups() else None
                            })
                            break
                            
            print(f"成功解析文件 {log_name} 的日志")
        except Exception as e:
            print(f"解析日志文件 {log_file} 时出错: {str(e)}")
            
    def analyze_all_logs(self):
        """分析所有日志文件"""
        self.matches = []  # 清空之前的匹配结果
        
        # 获取所有日志文件
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        print(f"找到的日志文件数量: {len(log_files)}")  # 添加调试信息
        
        # 分析每个日志文件
        for file_path in log_files:
            print(f"正在分析文件: {file_path}")  # 添加调试信息
            self.analyze_log_file(file_path)
        
        print(f"分析完成，总匹配数: {len(self.matches)}")  # 添加调试信息
            
    def prepare_chart_data(self):
        """准备图表数据"""
        if not self.matches:
            print("没有匹配结果")  # 添加调试信息
            return None
        
        # 创建数据列表
        data = []
        for match in self.matches:
            # 获取匹配信息
            rule_name = match['rule_name']
            timestamp = match['timestamp']
            line = match['line']
            matches = match['matches']
            file_path = match['file_path']
            line_number = match['line_number']
            is_show = match['is_show'] # 获取 is_show
            translate = match['translate'] # 获取 translate
            
            # 添加到数据列表
            data.append({
                'rule_name': rule_name,
                'timestamp': timestamp,
                'line': line,
                'matches': matches,
                'file_path': file_path,
                'line_number': line_number,
                'is_show': is_show, # 添加 is_show 到 DataFrame 数据
                'translate': translate # 添加 translate 到 DataFrame 数据
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        print(f"创建的DataFrame大小: {len(df)}")  # 添加调试信息
        return df
        
    def plot_matches(self):
        """绘制匹配规则统计图表"""
        df = self.prepare_chart_data()
        if df is None:
            print("没有数据可供绘图")
            return
            
        # 设置中文字体
        font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
        if not os.path.exists(font_path):
            font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
        if not os.path.exists(font_path):
            font_path = 'C:/Windows/Fonts/simsun.ttc'  # 宋体
        
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置图表风格
        plt.style.use('seaborn')
        
        # 创建图表 - 修改为3x2的布局
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
        
        # 1. 绘制规则匹配数量统计
        rule_counts = df['rule_name'].value_counts()
        rule_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('规则匹配数量统计', fontproperties=font_prop if 'font_prop' in locals() else None)
        ax1.set_xlabel('规则名称', fontproperties=font_prop if 'font_prop' in locals() else None)
        ax1.set_ylabel('匹配次数', fontproperties=font_prop if 'font_prop' in locals() else None)
        # 设置x轴标签字体
        ax1.set_xticklabels(ax1.get_xticklabels(), fontproperties=font_prop if 'font_prop' in locals() else None)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. 绘制时间序列图
        time_series = df.groupby([df['timestamp'].dt.date, 'rule_name']).size().unstack()
        time_series.plot(kind='line', ax=ax2)
        ax2.set_title('规则匹配时间趋势', fontproperties=font_prop if 'font_prop' in locals() else None)
        ax2.set_xlabel('日期', fontproperties=font_prop if 'font_prop' in locals() else None)
        ax2.set_ylabel('匹配次数', fontproperties=font_prop if 'font_prop' in locals() else None)
        # 设置图例字体
        ax2.legend(title='规则名称', bbox_to_anchor=(1.05, 1), loc='upper left', 
                  prop=font_prop if 'font_prop' in locals() else None)
        # 设置x轴标签字体
        ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=font_prop if 'font_prop' in locals() else None)
        
        # 3. 绘制日志文件分布
        file_counts = df['log_file'].value_counts()
        file_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
        ax3.set_title('日志文件分布', fontproperties=font_prop if 'font_prop' in locals() else None)
        # 设置饼图标签字体
        ax3.set_xticklabels(ax3.get_xticklabels(), fontproperties=font_prop if 'font_prop' in locals() else None)
        
        # 4. 绘制每个日志文件的规则分布
        file_rule_counts = df.groupby(['log_file', 'rule_name']).size().unstack()
        file_rule_counts.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('各日志文件规则分布', fontproperties=font_prop if 'font_prop' in locals() else None)
        ax4.set_xlabel('日志文件', fontproperties=font_prop if 'font_prop' in locals() else None)
        ax4.set_ylabel('匹配次数', fontproperties=font_prop if 'font_prop' in locals() else None)
        # 设置x轴标签字体
        ax4.set_xticklabels(ax4.get_xticklabels(), fontproperties=font_prop if 'font_prop' in locals() else None)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        # 设置图例字体
        ax4.legend(title='规则名称', bbox_to_anchor=(1.05, 1), loc='upper left',
                  prop=font_prop if 'font_prop' in locals() else None)
        
        # 5. 绘制用户登录统计
        # 提取用户登录数据
        login_data = df[df['rule_name'] == '用户登录'].copy()
        if not login_data.empty and login_data['matches'].notna().any():
            # 提取用户名
            login_data['username'] = login_data['matches'].apply(lambda x: x[0] if x else None)
            user_counts = login_data['username'].value_counts()
            
            # 绘制用户登录次数统计
            user_counts.plot(kind='bar', ax=ax5)
            ax5.set_title('用户登录次数统计', fontproperties=font_prop if 'font_prop' in locals() else None)
            ax5.set_xlabel('用户名', fontproperties=font_prop if 'font_prop' in locals() else None)
            ax5.set_ylabel('登录次数', fontproperties=font_prop if 'font_prop' in locals() else None)
            ax5.set_xticklabels(ax5.get_xticklabels(), fontproperties=font_prop if 'font_prop' in locals() else None)
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
            
            # 6. 绘制用户登录时间分布
            login_data['hour'] = login_data['timestamp'].dt.hour
            hourly_logins = login_data.groupby('hour').size()
            hourly_logins.plot(kind='line', marker='o', ax=ax6)
            ax6.set_title('用户登录时间分布', fontproperties=font_prop if 'font_prop' in locals() else None)
            ax6.set_xlabel('小时', fontproperties=font_prop if 'font_prop' in locals() else None)
            ax6.set_ylabel('登录次数', fontproperties=font_prop if 'font_prop' in locals() else None)
            ax6.set_xticks(range(0, 24))
            ax6.grid(True)
        else:
            ax5.text(0.5, 0.5, '没有用户登录数据', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax5.transAxes,
                    fontproperties=font_prop if 'font_prop' in locals() else None)
            ax6.text(0.5, 0.5, '没有用户登录数据',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax6.transAxes,
                    fontproperties=font_prop if 'font_prop' in locals() else None)
        
        # 调整布局
        plt.tight_layout()
        plt.show()

    def analyze_log_file(self, file_path):
        """分析单个日志文件"""
        try:
            print(f"开始分析文件: {file_path}")
            print(f"规则数量: {len(self.rules)}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 解析时间戳
                    timestamp = self.parse_timestamp(line)
                    if timestamp is None:
                        continue
                        
                    # 应用所有规则
                    for index, rule in self.rules.iterrows():  # 修改这里，使用iterrows()
                        matches = self.apply_rule(rule, line)
                        if matches:
                            self.matches.append({
                                'rule_name': rule['name'],
                                'timestamp': timestamp,
                                'line': line,
                                'matches': matches,
                                'file_path': file_path,
                                'line_number': line_number,
                                'is_show': rule['is_show'],  # 添加 is_show
                                'translate': rule['translate'] # 添加 translate
                            })
                            print(f"找到匹配: 规则={rule['name']}, 行号={line_number}")
            
            print(f"文件分析完成: {file_path}")
        except Exception as e:
            print(f"分析文件 {file_path} 时出错: {str(e)}")

    def parse_timestamp(self, line):
        """从日志行中解析时间戳"""
        try:
            # 尝试匹配时间戳格式：YYYY-MM-DD HH:MM:SS
            time_match = re.search(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', line)
            if time_match:
                timestamp = datetime.strptime(time_match.group(), '%Y-%m-%d %H:%M:%S')
                return timestamp
        except Exception as e:
            print(f"解析时间戳时出错: {str(e)}")
        return None

    def apply_rule(self, rule, line):
        """应用正则规则到日志行"""
        try:
            # 从DataFrame行中获取pattern
            pattern = rule['pattern'] if isinstance(rule, pd.Series) else rule.get('pattern')
            if not pattern:
                return None
            
            match = re.search(pattern, line)
            if match:
                return match.groups() if match.groups() else None
        except Exception as e:
            print(f"应用规则时出错: {str(e)}")
        return None

def main():
    # 使用示例
    analyzer = LogAnalyzer('rules.xlsx', 'logs')
    analyzer.load_rules()
    analyzer.analyze_all_logs()
    analyzer.plot_matches()

if __name__ == "__main__":
    main() 