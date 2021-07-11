import pickle
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from load_data import *
import argparse
import numpy as np
import random


# Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn function
    acc = accuracy_score(labels, preds)
    metric = {'accuracy': acc, }
    return metric


# Fix random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    # Fix seed
    seed_everything(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # load dataset
    train_dataset = load_data("./data/train/train.tsv")
    train_label = train_dataset['label'].values
    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting hyperparameter
    config = AutoConfig.from_pretrained(args.model_name)
    with open('./label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    config.num_labels = len(label_type)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(device)

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


def main(args):
    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model_name : ["xlm-roberta-large", "xlm-roberta-base", "bert-base-multilingual-cased"]
    parser.add_argument('--model_name', type=str)
    parser.add_argument(
        '--seed', type=int, default=2021
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results'
    )
    parser.add_argument(
        '--save_total_limit', type=int, default=3
    )
    parser.add_argument(
        '--save_steps', type=int, default=500
    )
    parser.add_argument(
        '--epochs', type=int, default=10
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5
    )
    parser.add_argument(
        '--batch_size', type=int, default=18
    )
    parser.add_argument(
        '--warmup_steps', type=int, default=300
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01
    )
    args = parser.parse_args()
    print(args)
    main(args)
