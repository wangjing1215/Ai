import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QScrollArea, QTabWidget, QComboBox,
                            QCheckBox, QGroupBox, QToolTip, QTextEdit, QTextBrowser,
                            QDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
import pyqtgraph as pg
import numpy as np
from log_analyzer import LogAnalyzer
import pandas as pd
import logging

class RuleConfigDialog(QDialog):
    def __init__(self, parent=None, rule_names=None, default_selected=None):
        super().__init__(parent)
        self.setWindowTitle("规则配置")
        self.setModal(True)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 添加全选和全不选按钮
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        deselect_all_btn = QPushButton("全不选")
        select_all_btn.clicked.connect(self.select_all)
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        layout.addLayout(button_layout)
        
        # 创建复选框列表
        self.checkboxes = {}
        for rule_name in rule_names:
            checkbox = QCheckBox(rule_name)
            checkbox.setChecked(default_selected.get(rule_name, True))  # 默认选中
            self.checkboxes[rule_name] = checkbox
            layout.addWidget(checkbox)
        
        # 添加确定和取消按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        cancel_button = QPushButton("取消")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def select_all(self):
        """全选所有规则"""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all(self):
        """取消全选所有规则"""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_rules(self):
        """获取选中的规则"""
        return [rule for rule, checkbox in self.checkboxes.items() if checkbox.isChecked()]

class LogAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_file = 'log_analyzer_config.json'
        self.load_config()  # 加载配置
        self.initUI()
        self.df = None  # 存储数据
        self.rule_names = []  # 存储规则名称
        
    def load_config(self):
        """加载配置文件"""
        self.last_rules_file = None
        self.last_logs_dir = None
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.last_rules_file = config.get('rules_file')
                    self.last_logs_dir = config.get('logs_dir')
            except Exception as e:
                print(f"加载配置文件时出错: {str(e)}")
                
    def save_config(self):
        """保存配置文件"""
        config = {
            'rules_file': self.rules_file,
            'logs_dir': self.logs_dir
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存配置文件时出错: {str(e)}")
            
    def initUI(self):
        # 设置PyQtGraph样式 - 移到最前面
        pg.setConfigOption('background', 'w')  # 白色背景
        pg.setConfigOption('foreground', 'k')  # 黑色前景
        pg.setConfigOption('antialias', True)  # 抗锯齿
        
        # 设置窗口标题和大小
        self.setWindowTitle('日志分析工具')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # 规则文件选择部分
        rules_layout = QHBoxLayout()
        self.rules_label = QLabel('规则文件：未选择')
        self.rules_btn = QPushButton('选择规则文件')
        self.rules_btn.clicked.connect(self.select_rules_file)
        rules_layout.addWidget(self.rules_label)
        rules_layout.addWidget(self.rules_btn)
        
        # 日志目录选择部分
        logs_layout = QHBoxLayout()
        self.logs_label = QLabel('日志目录：未选择')
        self.logs_btn = QPushButton('选择日志目录')
        self.logs_btn.clicked.connect(self.select_logs_dir)
        logs_layout.addWidget(self.logs_label)
        logs_layout.addWidget(self.logs_btn)
        
        # 分析按钮
        self.analyze_btn = QPushButton('开始分析')
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        
        # 添加控制面板组件
        control_layout.addLayout(rules_layout)
        control_layout.addLayout(logs_layout)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # 右侧图表显示区域
        self.tab_widget = QTabWidget()
        
        # 修改右侧布局为垂直布局
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # 添加控制面板
        control_group = QGroupBox("图表控制")
        control_layout = QVBoxLayout()
        
        # 添加规则设置按钮
        self.rule_config_btn = QPushButton("规则设置")
        self.rule_config_btn.clicked.connect(self.show_rule_config_dialog)
        
        # 添加控制组件到控制面板
        control_layout.addWidget(self.rule_config_btn)
        control_group.setLayout(control_layout)
        
        # 创建图表
        self.plot_widget = pg.PlotWidget()
        
        # 创建日志显示窗口
        self.log_browser = QTextBrowser()
        self.log_browser.setMaximumHeight(200)  # 设置最大高度
        self.log_browser.setReadOnly(True)  # 设置为只读
        
        # 添加组件到右侧布局
        right_layout.addWidget(control_group)
        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.log_browser)  # 添加日志显示窗口
        
        right_panel.setLayout(right_layout)
        
        # 修改主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)
        
        # 设置主布局
        central_widget.setLayout(main_layout)
        
        # 初始化文件路径
        self.rules_file = self.last_rules_file
        self.logs_dir = self.last_logs_dir
        
        # 更新标签显示
        if self.rules_file:
            self.rules_label.setText(f'规则文件：{os.path.basename(self.rules_file)}')
        if self.logs_dir:
            self.logs_label.setText(f'日志目录：{os.path.basename(self.logs_dir)}')
            
        # 检查是否可以启用分析按钮
        self.check_ready()
        
        
    def select_rules_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, '选择规则文件', 
            self.last_rules_file if self.last_rules_file else '', 
            'Excel Files (*.xlsx *.xls)')
        if file_name:
            self.rules_file = file_name
            self.rules_label.setText(f'规则文件：{os.path.basename(file_name)}')
            self.check_ready()
            self.save_config()  # 保存配置
            
    def select_logs_dir(self):
        dir_name = QFileDialog.getExistingDirectory(
            self, '选择日志目录',
            self.last_logs_dir if self.last_logs_dir else '')
        if dir_name:
            self.logs_dir = dir_name
            self.logs_label.setText(f'日志目录：{os.path.basename(dir_name)}')
            self.check_ready()
            self.save_config()  # 保存配置
            
    def check_ready(self):
        """检查是否所有必要的文件都已选择"""
        self.analyze_btn.setEnabled(
            bool(self.rules_file and self.logs_dir))
            
    def update_plot(self, selected_rules=None):
        """更新图表显示"""
        if self.df is None:
            return
            
        self.plot_widget.clear()
        
        # 获取选中的规则
        if selected_rules is None:
            selected_rules = self.rule_names  # 默认显示所有规则
        
        if len(selected_rules) == 0:
            return
            
        # 根据is_show过滤需要显示的规则
        rules_to_show = [rule for rule in selected_rules if self.df[self.df['rule_name'] == rule]['is_show'].iloc[0] == 'Y']

        if len(rules_to_show) == 0:
            return

        # 按时间排序数据
        self.df = self.df.sort_values('timestamp')
        
        # 为每个选中的规则绘制时间序列
        for rule_index, rule in enumerate(rules_to_show): # 遍历需要显示的规则
            # 获取该规则的所有匹配记录
            rule_data = self.df[self.df['rule_name'] == rule].copy() # 使用copy避免SettingWithCopyWarning
            
            if not rule_data.empty:
                # 获取时间戳和匹配值
                timestamps = rule_data['timestamp'].values
                x = timestamps.astype(np.int64) // 10**9  # 将numpy.datetime64转换为Unix时间戳
                
                # 获取翻译格式
                translate_format = self.df[self.df['rule_name'] == rule]['translate'].iloc[0]

                # 检查是否所有匹配值都是数值
                is_numeric = True
                y = np.ones(len(rule_data)) * (rule_index + 1)  # 每条曲线使用固定的y值，从1开始递增
                unique_states = set()  # 用于存储唯一的状态值
                
                for i, match in enumerate(rule_data['matches']):
                    if match:
                        try:
                            y[i] = float(match[0])
                        except ValueError:
                            is_numeric = False
                            unique_states.add(".".join(match))
                
                # 数值型结果：绘制曲线
                pen = pg.mkPen(width=2, color=pg.intColor(len(self.plot_widget.plotItem.items)))
                self.plot_widget.plot(x=x, y=y, name=rule, pen=pen)
                
                # 添加散点图
                scatter = pg.ScatterPlotItem(
                    x=x, y=y, 
                    symbol=['o', 's', 't', 'd', '+', 'x'][rule_index % 6],  # 不同曲线使用不同符号
                    size=15, 
                    name=rule,
                    hoverable=True,  # 启用悬停
                    hoverSymbol='s',  # 悬停时的符号
                    hoverSize=20,  # 悬停时的大小
                    hoverPen=pg.mkPen('r'),  # 悬停时的边框颜色
                    hoverBrush=pg.mkBrush('g')  # 悬停时的填充颜色
                )
                
                # 存储提示信息
                scatter.tooltips = []
                
                # 为每个点添加提示信息，只存储日志原文
                for i, line in enumerate(rule_data['line']):
                    scatter.tooltips.append(line)  # 只存储日志原文
                    
                    # 在点之间添加文字标注
                    if i > 0:  # 从第二个点开始
                        # 计算两个点之间的中点
                        mid_x = (x[i-1] + x[i]) / 2
                        mid_y = (y[i-1] + y[i]) / 2
                        
                        # 获取匹配值
                        match_value = ".".join(rule_data['matches'].iloc[i]) if rule_data['matches'].iloc[i] else ""

                        # 格式化标注文字
                        label_text = translate_format % match_value if "%s" in translate_format else translate_format

                        # 创建文字标注，位置向上偏移
                        text = pg.TextItem(
                            text=label_text,
                            color='k',
                            anchor=(0.5, 1.2)  # 修改anchor，使文字显示在点的上方
                        )
                        text.setPos(mid_x, mid_y)
                        self.plot_widget.addItem(text)
                
                # 定义点击回调函数
                def create_click_callback(scatter_item, click_time):
                    def on_click(_, points):
                        if len(points) > 0:
                            point = points[0]
                            index = point.data()
                            if hasattr(scatter_item, 'tooltips') and index < len(scatter_item.tooltips):
                                # 获取点击的时间戳
                                click_timestamp = click_time[index]
                                
                                # 收集所有曲线在该时间点之前的最近日志
                                all_logs = []
                                # 首先添加当前点击的日志
                                current_rule = scatter_item.name()
                                current_log = scatter_item.tooltips[index]
                                
                                # 然后添加其他规则的日志
                                for rule in selected_rules:
                                    rule_data = self.df[self.df['rule_name'] == rule]
                                    if not rule_data.empty:
                                        # 找到点击时间之前的所有时间点
                                        rule_timestamps = rule_data['timestamp'].values
                                        # 使用numpy的比较操作
                                        before_mask = np.array([ts <= click_timestamp for ts in rule_timestamps])
                                        before_times = rule_timestamps[before_mask]
                                        
                                        if len(before_times) > 0:
                                            # 获取最近的时间点
                                            closest_time = before_times[-1]
                                            closest_idx = np.where(rule_timestamps == closest_time)[0][0]
                                            log_text = rule_data.iloc[closest_idx]['line']
                                                # 获取匹配值
                                            match_value = ".".join(rule_data.iloc[closest_idx]['matches']) if rule_data.iloc[closest_idx]['matches'] else ""
                                            translate_format = rule_data.iloc[closest_idx]["translate"]
                                            # 格式化标注文字
                                            label_text = translate_format % match_value if "%s" in translate_format else translate_format
                                            if rule != current_rule:  # 跳过当前规则
                                                all_logs.append(f"【{rule}】\n{log_text}|||{label_text}\n")
                                            else:
                                                all_logs.insert(0, f"【{rule}】\n{log_text}|||{label_text}\n")
                                
                                # 更新日志显示窗口
                                self.log_browser.setText("\n".join(all_logs))
                    return on_click
                
                # 连接点击信号
                scatter.sigClicked.connect(create_click_callback(scatter, timestamps))
                
                # 为每个点设置索引
                scatter.setData(x=x, y=y, data=list(range(len(x))))
                
                self.plot_widget.addItem(scatter)
        
        # 添加图例
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.setLabel('left', '值/状态')
        self.plot_widget.setLabel('bottom', '时间')
        
        # 设置时间轴
        self.plot_widget.getAxis('bottom').setStyle(showValues=True)
        
        # 设置时间轴刻度
        unique_times = self.df['timestamp'].unique()
        time_span = (unique_times[-1] - unique_times[0]).total_seconds()
        if time_span <= 3600:
            interval = 300
        elif time_span <= 86400:
            interval = 3600
        else:
            interval = 86400
        
        ticks = []
        current_time = pd.Timestamp(unique_times[0])
        end_time = pd.Timestamp(unique_times[-1])
        while current_time <= end_time:
            ticks.append((current_time.timestamp(), 
                         current_time.strftime('%Y-%m-%d %H:%M')))
            current_time += pd.Timedelta(seconds=interval)
        
        self.plot_widget.getAxis('bottom').setTicks([ticks])
        
        # 自动调整显示范围
        self.plot_widget.autoRange()
        
        # 添加悬浮提示
        self.plot_widget.setToolTip("")
        try:
            self.plot_widget.scene().sigMouseMoved.disconnect()
        except:
            pass
        
    def show_rule_config_dialog(self):
        """显示规则配置弹窗"""
        if len(self.rule_names) == 0:
            QMessageBox.warning(self, "警告", "请先加载规则文件")
            return
        
        # 获取所有规则的当前选中状态
        current_selected = {}
        for rule_name in self.rule_names:
            current_selected[rule_name] = True  # 默认选中
        
        # 创建并显示配置弹窗
        dialog = RuleConfigDialog(self, self.rule_names, current_selected)
        if dialog.exec_() == QDialog.Accepted:
            # 获取选中的规则
            selected_rules = dialog.get_selected_rules()
            
            # 更新图表
            self.update_plot(selected_rules)

    def start_analysis(self):
        """开始分析日志"""
        try:
            analyzer = LogAnalyzer(self.rules_file, self.logs_dir)
            analyzer.load_rules()
            analyzer.analyze_all_logs()
            
            # 添加调试信息
            print(f"规则文件: {self.rules_file}")
            print(f"日志目录: {self.logs_dir}")
            print(f"匹配结果数量: {len(analyzer.matches)}")
            
            self.df = analyzer.prepare_chart_data()
            
            if self.df is not None:
                # 添加调试信息
                print(f"DataFrame 大小: {len(self.df)}")
                print(f"DataFrame 列: {self.df.columns.tolist()}")
                print(f"DataFrame 前几行:\n{self.df.head()}")
                
                # 获取规则名称
                self.rule_names = self.df['rule_name'].unique()
                # 更新图表
                self.update_plot()
            else:
                QMessageBox.warning(self, '警告', '没有数据可供绘图')
                
        except Exception as e:
            error_msg = f'分析过程中出现错误：{str(e)}'
            print(error_msg)
            logging.exception(error_msg)
            QMessageBox.critical(self, '错误', error_msg)

def main():
    app = QApplication(sys.argv)
    ex = LogAnalyzerGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 