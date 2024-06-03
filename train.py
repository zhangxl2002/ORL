import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# import pytorch_lightning as pl


from datasets import load_dataset

from datasets import load_metric

import matplotlib.pyplot as plt

import sys
import datetime

import shutil

from tqdm import tqdm

from utils import *
from MPQADataset import MPQADataset
from T5FineTuner import *
from Datapool import *

def train():
    # 不使用AL的训练流程

    # 设置参数
    args_dict = dict(
        data_dir="zhangxl2002/mpqa_ORL",
        model_name_or_path='t5-base',
        tokenizer_name_or_path='t5-base',
        max_seq_length=256,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        eval_batch_size=8,
        num_train_epochs=10,
        gradient_accumulation_steps=16,
        n_gpu=1,
        seed=42,
    )
    args = argparse.Namespace(**args_dict)

    set_seed(42)
    
    # 加载数据集
    dataset = load_dataset("zhangxl2002/mpqa_ORL")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # 加载模型
    model = T5FineTuner(args)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{current_time}.log"

    max_epochs = 20
    current_strategy = "FULL"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = model.train_dataloader()
    test_dataloader = model.test_dataloader()
    
    print_log = True
    write_log = False
    save_model = False

    model.model.to(device)
    if write_log:
        sys.stdout = open('log/'+log_file_name, 'w+')

    full_train_loss_record = np.zeros(max_epochs + 1)
    full_test_loss_record = np.zeros(max_epochs + 1)
    full_test_F1_overall_record= np.zeros(max_epochs + 1)
    full_test_F1_AGENT_record = np.zeros(max_epochs + 1)
    full_test_F1_TARGET_record = np.zeros(max_epochs + 1)

    max_overall_f1 = 0.0
    for epoch in range(max_epochs):
        # 训练
        total_loss = 0.0
        num_batches = 0
        model.model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {key: value.to(device) for key, value in batch.items()} 
            loss = model.training_step(batch, i)
            loss.backward()
            total_loss += loss.item()
            num_batches += 1
            model.optimizer_step()
        avg_loss = total_loss / num_batches
        if print_log: 
            print(f"--------------Epoch [{epoch + 1}/{max_epochs}]---------------")
            print(f"Average Loss: {avg_loss:.4f}")
            print("-------------------------------------------------------------")


        # 每一个epoch训练完成后，在验证集上计算损失 计算token级别的precision,recall,F1
        test_loss = 0.0
        num_test_batches = 0

        true_labels = []
        pred_labels = []

        test_true_labels = []
        test_pred_labels = []
        model.model.eval()
        for i, test_batch in tqdm(enumerate(test_dataloader)):
            test_batch = {key: value.to(device) for key, value in test_batch.items()} 
            test_loss += model.validation_step(test_batch, i).item()
            num_test_batches += 1
        for i, test_batch in tqdm(enumerate(test_dataloader)):
            # precision,recall,F1
            test_batch = {key: value.to(device) for key, value in test_batch.items()} 
            input_ids = test_batch['source_ids']
            attention_mask = test_batch['source_mask']
            test_outs = model.model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_length=500,
                                        num_beams=2)
            test_dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip() for ids in test_outs]
            test_target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                        for ids in test_batch["target_ids"]]
            test_texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                        for ids in test_batch["source_ids"]]
            true_label = [generate_label(test_texts[i].strip(), test_target[i].strip()) if test_target[i].strip() != 'none' else [
                "O"]*len(test_texts[i].strip().split()) for i in range(len(test_texts))]
            pred_label = [generate_label(test_texts[i].strip(), test_dec[i].strip()) if test_dec[i].strip() != 'none' else [
                "O"]*len(test_texts[i].strip().split()) for i in range(len(test_texts))]
            test_true_labels.extend(true_label)
            test_pred_labels.extend(pred_label)
        model.model.train()

        metric = load_metric("seqeval")
        test_metric_result = metric.compute(predictions=test_pred_labels, references=test_true_labels)
        
        if test_metric_result['overall_f1'] > max_overall_f1:
            max_overall_f1 = test_metric_result['overall_f1']
            print("best_overall_f1 update:",max_overall_f1)
            if save_model == True:
                new_folder = f"model_F1_{max_overall_f1:.5f}_epoch_{epoch}"
                base_dir = f"./saved_models/strategy_FULL"
                full_dir = base_dir + "/" + new_folder
                model.model.save_pretrained(full_dir)
                print(f"Model saved at {full_dir}")
                files = os.listdir(base_dir)
                for file in files:
                    if file == new_folder:
                        continue
                    old_model_path = os.path.join(base_dir, file)
                    if os.path.isdir(old_model_path) and file.startswith("model"):
                        shutil.rmtree(old_model_path)
                        print(f"delete:{old_model_path}")

        avg_test_loss = test_loss / num_test_batches
        if print_log: 
            print("-----------------------------------------------------------")
            print("strategy:", current_strategy)
            print("epoch:", epoch)

            print(f"Test Loss: {avg_test_loss:.4f}")
            print(f"F1_overall: {test_metric_result['overall_f1']:.4f}")
            print(f"F1_AGENT: {test_metric_result['AGENT']['f1']:.4f}")
            print(f"F1_TARGET: {test_metric_result['TARGET']['f1']:.4f}")
            print("-----------------------------------------------------------")

        full_train_loss_record[epoch] = avg_loss

        full_test_loss_record[epoch] = avg_test_loss
        full_test_F1_overall_record[epoch] = test_metric_result['overall_f1']
        full_test_F1_AGENT_record[epoch] = test_metric_result['AGENT']['f1']
        full_test_F1_TARGET_record[epoch] = test_metric_result['TARGET']['f1']
        if save_model == True:
            save_array(current_strategy+"_test_loss_record", full_test_loss_record)
            save_array(current_strategy+"_test_F1_overall_record", full_test_F1_overall_record)
            save_array(current_strategy+"_test_F1_TARGET_record", full_test_F1_TARGET_record)
            save_array(current_strategy+"_test_F1_AGENT_record", full_test_F1_AGENT_record)
    if write_log:
        sys.stdout.close()
def trainWithAL():
    # 使用AL的做法

    # 设置参数
    args_dict = dict(
        data_dir="zhangxl2002/mpqa_ORL",
        model_name_or_path='t5-base',
        tokenizer_name_or_path='t5-base',
        max_seq_length=256,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        eval_batch_size=8,
        num_train_epochs=10,
        gradient_accumulation_steps=16,
        n_gpu=1,
        seed=42,
    )
    args = argparse.Namespace(**args_dict)

    set_seed(42)
    # 加载数据集
    dataset = load_dataset("zhangxl2002/mpqa_ORL")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # 加载模型
    model = T5FineTunerWithAL(args)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{current_time}.log"

    write_log = True

    # 将标准输出重定向到文件
    if write_log:
        sys.stdout = open('log/'+log_file_name, 'w+')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_epochs = 20
    end_loss = 0.025

    max_iters = 12 # 每个iterator选择新的数据加入已标注数据的集合并重新训练模型

    # 初始化，默认初始标注数据占总数量的5%，使用RANDOM选择策略
    model = T5FineTunerWithAL(args) 

    # 验证集始终是一样的
    test_dataloader = model.test_dataloader()
    # test_dataloader = model.test_dataloader()

    strategies = ["BEAM"]

    print_log = True
    save_model = False

    # 记录loss和准确率
    # loss：键为(strategy,iter,epoch 三元组 值为train_loss或者val_loss
    train_loss_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    val_loss_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    val_F1_overall_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    val_F1_AGENT_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    val_F1_TARGET_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))

    test_loss_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))

    test_F1_overall_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    test_F1_AGENT_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    test_F1_TARGET_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))

    test_precision_overall_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    test_precision_AGENT_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    test_precision_TARGET_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))

    test_recall_overall_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    test_recall_AGENT_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))
    test_recall_TARGET_record = np.zeros((len(strategies) + 1, max_iters + 1, max_epochs + 1))

    for strategy_id in range(len(strategies)):
        current_strategy = strategies[strategy_id] 
        if print_log: print("strategy:", current_strategy)

        # 当前的策略，能够达到的最好的f1分数
        max_overall_f1 = 0.0 
        for iter in range(max_iters):
            if print_log: print("iter:", iter)
            # 对每轮新的数据都重新训练
            model.resetModel()
            if print_log: model.datapool.showDetail()
            model.model.to(device)

            # 训练集每轮更新，需要重新加载
            annotated_dataloader = model.datapool.getAnnotatedDataloader()

            for epoch in range(max_epochs):
                # 训练
                total_loss = 0.0
                num_batches = 0

                for i, batch in enumerate(annotated_dataloader):
                    batch = {key: value.to(device) for key, value in batch.items()} 
                    loss = model.training_step(batch, i)
                    loss.backward()
                    total_loss += loss.item()
                    num_batches += 1
                    model.optimizer_step()
                avg_loss = total_loss / num_batches
                if print_log: print(f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}")


                val_loss = 0.0
                num_val_batches = 0

                test_loss = 0.0
                num_test_batches = 0

                true_labels = []
                pred_labels = []

                test_true_labels = []
                test_pred_labels = []
                model.model.eval()

                for i, test_batch in enumerate(test_dataloader):
                    # precision,recall,F1
                    test_batch = {key: value.to(device) for key, value in test_batch.items()} 
                    input_ids = test_batch['source_ids']
                    attention_mask = test_batch['source_mask']
                    test_outs = model.model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                max_length=500,
                                                num_beams=2,
                                                length_penalty=0.0)
                    test_dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip() for ids in test_outs]
                    test_target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                                for ids in test_batch["target_ids"]]
                    test_texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                                for ids in test_batch["source_ids"]]
                    true_label = [generate_label(test_texts[i].strip(), test_target[i].strip()) if test_target[i].strip() != 'none' else [
                        "O"]*len(test_texts[i].strip().split()) for i in range(len(test_texts))]
                    pred_label = [generate_label(test_texts[i].strip(), test_dec[i].strip()) if test_dec[i].strip() != 'none' else [
                        "O"]*len(test_texts[i].strip().split()) for i in range(len(test_texts))]
                    test_true_labels.extend(true_label)
                    test_pred_labels.extend(pred_label)
                    del test_batch
                model.model.train()

                metric = load_metric("seqeval")

                test_metric_result = metric.compute(predictions=test_pred_labels, references=test_true_labels)

                if print_log: 
                    print("-----------------------------------------------------------")
                    print("strategy:", current_strategy)
                    print("iter:", iter)
                    print("epoch:", epoch)

                    print(f"F1_overall: {test_metric_result['overall_f1']:.4f}")
                    print(f"F1_AGENT: {test_metric_result['AGENT']['f1']:.4f}")
                    print(f"F1_TARGET: {test_metric_result['TARGET']['f1']:.4f}")

                    print(f"precision_overall: {test_metric_result['overall_precision']:.4f}")
                    print(f"precision_AGENT: {test_metric_result['AGENT']['precision']:.4f}")
                    print(f"precision_TARGET: {test_metric_result['TARGET']['precision']:.4f}")

                    print(f"recall_overall: {test_metric_result['overall_recall']:.4f}")
                    print(f"recall_AGENT: {test_metric_result['AGENT']['recall']:.4f}")
                    print(f"recall_TARGET: {test_metric_result['TARGET']['recall']:.4f}")
                    print("-----------------------------------------------------------")

                if test_metric_result['overall_f1'] > max_overall_f1:
                    max_overall_f1 = test_metric_result['overall_f1']
                    print("best_overall_f1 update:",max_overall_f1)
                    if save_model == True:
                        new_folder = f"model_F1_{max_overall_f1:.5f}_iter_{iter}_epoch_{epoch}"
                        base_dir = f"./saved_models/strategy_{current_strategy}"
                        full_dir = base_dir + "/" + new_folder
                        model.model.save_pretrained(full_dir)
                        print(f"Model saved at {full_dir}")
                        files = os.listdir(base_dir)
                        for file in files:
                            if file == new_folder:
                                continue
                            old_model_path = os.path.join(base_dir, file)
                            if os.path.isdir(old_model_path) and file.startswith("model"):
                                shutil.rmtree(old_model_path)
                                print(f"delete:{old_model_path}")
                        
                test_F1_overall_record[strategy_id][iter][epoch] = test_metric_result['overall_f1']
                test_F1_AGENT_record[strategy_id][iter][epoch] = test_metric_result['AGENT']['f1']
                test_F1_TARGET_record[strategy_id][iter][epoch] = test_metric_result['TARGET']['f1']

                test_precision_overall_record[strategy_id][iter][epoch] = test_metric_result['overall_precision']
                test_precision_AGENT_record[strategy_id][iter][epoch] = test_metric_result['AGENT']['precision']
                test_precision_TARGET_record[strategy_id][iter][epoch] = test_metric_result['TARGET']['precision']

                test_recall_overall_record[strategy_id][iter][epoch] = test_metric_result['overall_recall']
                test_recall_AGENT_record[strategy_id][iter][epoch] = test_metric_result['AGENT']['recall']
                test_recall_TARGET_record[strategy_id][iter][epoch] = test_metric_result['TARGET']['recall']
                
                if save_model == True:
                    save_array(current_strategy+"_test_F1_overall_record", test_F1_overall_record[strategy_id])
                    save_array(current_strategy+"_test_F1_TARGET_record", test_F1_TARGET_record[strategy_id])
                    save_array(current_strategy+"_test_F1_AGENT_record", test_F1_AGENT_record[strategy_id])

                    save_array(current_strategy+"_test_precision_overall_record", test_precision_overall_record[strategy_id])
                    save_array(current_strategy+"_test_precision_TARGET_record", test_precision_TARGET_record[strategy_id])
                    save_array(current_strategy+"_test_precision_AGENT_record", test_precision_AGENT_record[strategy_id])

                    save_array(current_strategy+"_test_recall_overall_record", test_recall_overall_record[strategy_id])
                    save_array(current_strategy+"_test_recall_TARGET_record", test_recall_TARGET_record[strategy_id])
                    save_array(current_strategy+"_test_recall_AGENT_record", test_recall_AGENT_record[strategy_id])

                if avg_loss < end_loss:
                    break

            # 所有epoch结束后，对未标注数据采用选择策略，选出新的标注数据，更新datapool
            model.update_datapool(strategy=current_strategy,add_percentage=0.05)
            if print_log: 
                print("update datapool!")
                model.datapool.showDetail()
        model.resetDatapool()
    if write_log:
        sys.stdout.close()

if __name__ == "__main__":    
    # trainWithAL()
    train()