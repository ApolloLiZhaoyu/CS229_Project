import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


if __name__ == '__main__':
    from transformers import AlbertTokenizer, AlbertForSequenceClassification
    import torch

    torch.set_num_threads(8)        

    checkpoint = 'path/to/your/model'
    tokenizer = AlbertTokenizer.from_pretrained(checkpoint)
    model = AlbertForSequenceClassification.from_pretrained(checkpoint)

    task_path = 'glue_data/CoLA/test.tsv'

    sents = []
    outputs = []

    with open(task_path, 'r') as f:
        f.readline()
        for line in f:
            sent = line.strip('\n').split('\t')[1]
            sents.append(sent)

    for idx, sent in enumerate(sents):
        print(idx)
        input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)  # Batch size 1
        output = model(input_ids)
        outputs.append(output[0][0].data)

    with open('result.tsv', 'w+') as f:
        for idx, output in enumerate(outputs):
            f.write('{}\t{}\t{}\t{}\n'.format(idx, output.argmax(), output[0], output[1]))
