import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class GenDateSet:
    def __init__(self, tokenizer, train_file, val_file, max_length=128, batch_size=10):
        self.train_file = train_file
        self.val_file = val_file
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def gen_data(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError(f"Dataset not found: {file}")

        input_ids, input_types, input_masks, labels = [], [], [], []

        with open(file, encoding='utf8') as f:
            data = json.load(f)

        if not data:
            raise ValueError("Dataset is empty")
        data = data[:100]
        for index, item in enumerate(data):
            print(item)
            text = item['content']
            tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)
            input_ids.append(tokens['input_ids'])
            input_types.append(tokens['token_type_ids'])
            input_masks.append(tokens['attention_mask'])
            a = torch.zeros(217)
            a[random.randint(0, 216)] = 1
            labels.append(a)

            if index % 1000 == 0:
                print(f'Processed {index} items')

        data_gen = TensorDataset(
            torch.LongTensor(np.array(input_ids)),
            torch.LongTensor(np.array(input_types)),
            torch.LongTensor(np.array(input_masks)),
            torch.stack(labels)
        )

        sampler = RandomSampler(data_gen)
        return DataLoader(data_gen, sampler=sampler, batch_size=self.batch_size)

    def gen_train_data(self):
        return self.gen_data(self.train_file)

    def gen_val_data(self):
        return self.gen_data(self.val_file)


def val(model, device, data):
    model.eval()
    acc = 0
    total = 0
    for (input_id, types, masks, y) in tqdm(data):
        input_id, types, masks, y = input_id.to(device), types.to(device), masks.to(device), y.to(device)
        with torch.no_grad():
            # 获取模型预测
            pred = model(input_id, token_type_ids=types, attention_mask=masks).logits
            # 将预测值转换为二进制标签
            pred = torch.sigmoid(pred)  # 将 logits 转换为概率
            pred = (pred > 0.8).float()  # 将概率转换为二进制标签
            # 计算准确率
            acc += pred.eq(y).sum().item()
            total += y.numel()  # 计算总的标签数量

    return acc / total  # 返回准确率


def main():
    model_dir = r'D:\code\agent\pythonProject3\Ai\chinese-roberta-wwm-ext'
    train_file = 'data/train/usual_train.txt'
    val_file = 'data/eval/virus_eval_labeled.txt'
    save_model_path = './model/'

    max_length = 512
    batch_size = 10
    epochs = 10

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=217)

    dataset = GenDateSet(tokenizer, train_file, val_file, max_length, batch_size)
    train_data = dataset.gen_train_data()
    val_data = dataset.gen_val_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_acc = 0.0
    for epoch_index in range(epochs):
        model.train()
        for batch_epoch, (input_id, types, masks, y) in enumerate(tqdm(train_data)):
            input_id, types, masks, y = input_id.to(device), types.to(device), masks.to(device), y.to(device)

            outputs = model(input_id, token_type_ids=types, attention_mask=masks, labels=y)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_epoch % 10 == 0:
                print(f'Train Epoch: {epoch_index}, Batch: {batch_epoch}, Loss: {loss.item()}')

        acc = val(model, device, val_data)
        print(f'Train Epoch: {epoch_index}, Validation Accuracy: {acc:.4f}')

        if acc > best_acc:
            # 在保存之前确保所有参数都是连续的
            for name, param in model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            model.save_pretrained(save_model_path)
            tokenizer.save_pretrained(save_model_path)
            best_acc = acc


if __name__ == '__main__':
    main()
