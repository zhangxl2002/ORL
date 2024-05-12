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

def test(args):
    args_dict = dict(
        data_dir="zhangxl2002/mpqa_ORL", # path for data files
        output_dir="", # path to save the checkpoints
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
        early_stop_callback=False,
        fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )
    args = argparse.Namespace(**args_dict)

    set_seed(42)
    print(strategy)
    # 加载数据集
    dataset = load_dataset("zhangxl2002/mpqa_ORL")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5FineTuner(args)
    test_dataset = MPQADataset(tokenizer=tokenizer, dataset=dataset, type_path='test')
    test_loader = DataLoader(test_dataset, batch_size=32,
                                num_workers=2, shuffle=True)
    model.model.eval()
    model.model = model.model.to("cuda")
    outputs = []
    targets = []
    all_text = []
    true_labels = []
    pred_labels = []
    for batch in tqdm(test_loader):
        input_ids = batch['source_ids'].to("cuda")
        attention_mask = batch['source_mask'].to("cuda")
        outs = model.model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_length=500,
                                    length_penalty = 0.0,
                                    num_beams=2)
        dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip() for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                    for ids in batch["target_ids"]]
        texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                    for ids in batch["source_ids"]]
        true_label = [generate_label(texts[i].strip(), target[i].strip()) if target[i].strip() != 'none' else [
            "O"]*len(texts[i].strip().split()) for i in range(len(texts))]
        pred_label = [generate_label(texts[i].strip(), dec[i].strip()) if dec[i].strip() != 'none' else [
            "O"]*len(texts[i].strip().split()) for i in range(len(texts))]

        outputs.extend(dec)
        targets.extend(target)
        true_labels.extend(true_label)
        pred_labels.extend(pred_label)
        all_text.extend(texts)
    metric = load_metric("seqeval")

    for i in range(10):
        print(f"Text:  {all_text[i]}")
        print(f"Predicted Token Class:  {pred_labels[i]}")
        print(f"True Token Class:  {true_labels[i]}")
        print("=====================================================================\n")

    print(metric.compute(predictions=pred_labels, references=true_labels))


if __name__ == "__main__":    
    test()