import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import random
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 定义8个方面的类别
ASPECT_CATEGORIES = [
    '音质', '续航', '舒适度', '通话质量',
    '蓝牙连接', '性价比', '做工质量', '其他'
]

class BluetoothHeadphoneDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=88, is_training=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # 数据增强
        self.augment_prob = 0.3 if is_training else 0.0  # 只在训练时进行数据增强

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 随机数据增强
        if self.is_training and random.random() < self.augment_prob:
            # 随机删除一些字符
            words = list(text)
            num_to_delete = random.randint(1, max(1, len(words) // 10))
            for _ in range(num_to_delete):
                if words:
                    del words[random.randint(0, len(words) - 1)]
            text = ''.join(words)
        
        encoding = self.tokenizer(text,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_length,
                                return_tensors='pt')
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attention_output))
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = self.norm(x + self.dropout(ff_output))
        return x

class AspectClassifier(nn.Module):
    def __init__(self, num_aspects=8):
        super(AspectClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('/root/bert-base-chinese')
        self.bert.config.hidden_dropout_prob = 0.2  # 增加dropout
        self.bert.config.attention_probs_dropout_prob = 0.2
        hidden_size = self.bert.config.hidden_size
        
        # 注意力层和前馈网络
        self.attention = SelfAttention(hidden_size)
        self.feed_forward = FeedForward(hidden_size)
        
        # 增加一个分类器层
        self.classifier_layers = nn.ModuleList([
            nn.Linear(hidden_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)  # 新增的层
        ])
        
        self.final_classifier = nn.Linear(64, num_aspects)  # 修改输出层大小
        self.dropout = nn.Dropout(0.1)
        self.additional_dropout = nn.Dropout(0.1)  # 新增的dropout层
        self.activation = nn.ReLU()  # 改为ReLU激活函数
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        # 初始化所有线性层
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.classifier_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.final_classifier.weight)
        nn.init.zeros_(self.final_classifier.bias)

    def forward(self, input_ids, attention_mask):
        # BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后四层的hidden states
        last_four_layers = torch.stack(outputs.hidden_states[-4:], dim=1)
        sequence_output = torch.mean(last_four_layers, dim=1)
        
        # 添加注意力机制
        sequence_output = self.attention(sequence_output)
        sequence_output = self.feed_forward(sequence_output)
        
        # 平均池化
        pooled_output = torch.mean(sequence_output, dim=1)
        pooled_output = self.layer_norm(pooled_output)
        
        # 多层分类器与残差连接
        x = pooled_output
        for layer in self.classifier_layers:
            residual = x
            x = self.dropout(x)
            x = layer(x)
            x = self.activation(x)
            if x.size() == residual.size():
                x = x + residual
            x = self.additional_dropout(x)  # 新增的dropout层
        
        # 最终分类
        logits = self.final_classifier(x)
        return self.sigmoid(logits)

class AspectTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train(self, train_dataloader, val_dataloader=None,
              epochs=15, learning_rate=2e-5):
        # 初始化优化器和学习率调度器
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.1,  # 增大权重衰减
            eps=1e-8  # 提高数值稳定性
        )
        num_training_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 5,  # 增加warmup步数
            num_training_steps=num_training_steps
        )
        
        # 使用带权重的BCE损失
        pos_weight = torch.ones([8]).to(self.device) * 2  # 每个类别的正样本权重为2
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1}, Average loss: {avg_loss:.4f}')
            
            if val_dataloader:
                self.evaluate(val_dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        criterion = nn.BCELoss()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = (outputs > 0.3).int()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # 计算每个类别的指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 计算总体指标
        accuracy = np.mean(all_predictions == all_labels)
        
        # 计算每个类别的F1分数
        f1_scores = []
        for i in range(len(ASPECT_CATEGORIES)):
            true_pos = np.sum((all_predictions[:, i] == 1) & (all_labels[:, i] == 1))
            false_pos = np.sum((all_predictions[:, i] == 1) & (all_labels[:, i] == 0))
            false_neg = np.sum((all_predictions[:, i] == 0) & (all_labels[:, i] == 1))
            
            precision = true_pos / (true_pos + false_pos + 1e-10)
            recall = true_pos / (true_pos + false_neg + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)
        
        print(f'Validation loss: {avg_loss:.4f}')
        print(f'Overall accuracy: {accuracy:.4f}')
        print('\nF1 scores for each aspect:')
        for aspect, f1 in zip(ASPECT_CATEGORIES, f1_scores):
            print(f'{aspect}: {f1:.4f}')
        print(f'Average F1 score: {np.mean(f1_scores):.4f}')
        
        return avg_loss, accuracy, np.mean(f1_scores)
    
    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer(text,
                                truncation=True,
                                padding='max_length',
                                max_length=88,  # 使用新的最大长度
                                return_tensors='pt')
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # 获取原始输出和二值化预测
        raw_outputs = outputs.cpu().numpy()[0]
        predictions = (outputs > 0.5).int().cpu().numpy()[0]  # 提高阈值到0.5
        
        # 计算每个方面的置信度
        confidences = np.abs(raw_outputs - 0.5) * 2  # 将输出转换为0-1的置信度
        
        return predictions, confidences

    def evaluate_prediction(self, text, expected_aspects=None):
        """评估预测结果并给出评分"""
        predictions, confidences = self.predict(text)
        
        # 计算每个方面的评分
        aspect_scores = []
        total_mentioned = 0
        correct_predictions = 0
        
        print(f"\n输入文本: {text}")
        print("涉及方面及置信度:")
        
        for i, (aspect, pred, conf) in enumerate(zip(ASPECT_CATEGORIES, predictions, confidences)):
            if pred == 1:
                total_mentioned += 1
                score = conf * 100  # 转换为百分制
                aspect_scores.append(score)
                
                # 如果提供了期望的方面，检查是否正确
                if expected_aspects is not None:
                    if i in expected_aspects:
                        correct_predictions += 1
                        result = "✅"  # 正确
                    else:
                        result = "❌"  # 错误
                else:
                    result = ""
                
                print(f"  - {aspect}: {score:.1f}% {result}")
        
        # 计算总体评分
        if total_mentioned > 0:
            average_score = sum(aspect_scores) / len(aspect_scores)
            if expected_aspects is not None:
                accuracy = correct_predictions / max(total_mentioned, len(expected_aspects)) * 100
                print(f"\n预测准确率: {accuracy:.1f}%")
            print(f"平均置信度: {average_score:.1f}%")
            return accuracy if expected_aspects is not None else average_score
        else:
            print("没有检测到任何相关方面")
            return 0 if expected_aspects is not None else None

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('/root/bert-base-chinese')
    
    # 加载训练数据
    texts = []
    labels = []
    
    # 从 JSONL 文件读取数据
    with open('/root/code/bert/data/train/reviews_temp.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
            labels.append(data['label'])
    
    print(f"加载了 {len(texts)} 条评论数据")
    
    # 数据集划分
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {len(X_train)} 条")
    print(f"验证集: {len(X_val)} 条")
    
    # 创建数据集
    train_dataset = BluetoothHeadphoneDataset(
        X_train, y_train, tokenizer, max_length=88  # 调整为新的最大长度
    )
    val_dataset = BluetoothHeadphoneDataset(
        X_val, y_val, tokenizer, max_length=88
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True  # 减小 batch size 以提高稳定性
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16
    )
    
    # 初始化模型
    model = AspectClassifier()
    trainer = AspectTrainer(model, tokenizer, device)
    
    # 训练模型
    trainer.train(
        train_dataloader,
        val_dataloader,
        epochs=20,  # 增加训练轮数
        learning_rate=5e-6  # 进一步降低学习率
    )
    
    # 测试集
    test_cases = [
        {
            'text': "音质清晰，续航10小时，1500元性价比高",
            'aspects': [0, 1, 5]  # 音质、续航、性价比
        },
        {
            'text': "蓝牙稳定，延迟30毫秒，但通话有噪音",
            'aspects': [3, 4]  # 通话质量、蓝牙连接
        },
        {
            'text': "佩戴舒服，做工精细，2000元值得买",
            'aspects': [2, 5, 6]  # 舒适度、性价比、做工质量
        },
        {
            'text': "续航差，充电慢，但音质还行",
            'aspects': [0, 1]  # 音质、续航
        },
        {
            'text': "信号距离10米，降噪效果好",
            'aspects': [0, 4]  # 音质、蓝牙连接
        }
    ]
    
    print("\
=== 测试集预测结果 ===")
    total_accuracy = 0
    for test_case in test_cases:
        accuracy = trainer.evaluate_prediction(test_case['text'], test_case['aspects'])
        if accuracy is not None:
            total_accuracy += accuracy
    
    print(f"整体平均准确率: {total_accuracy / len(test_cases):.1f}%")

if __name__ == '__main__':
    main()