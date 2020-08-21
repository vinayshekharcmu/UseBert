import argparse
from torch.utils.data import DataLoader
from data.data_loader import NLIDataset
from functools import partial
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import glue_convert_examples_to_features
from config import *
import torch
import torch.nn as nn
from model.NLIClassifier import NLIClassifier
from tqdm import tqdm

class Example:
    def __init__(self, example_a, example_b, gold_label):
        self.text_a = example_a
        self.text_b = example_b
        self.label = LABEL_MAP.get(gold_label, LABEL_MAP[ENTAILMENT])
        self.guid = "GUID"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    return args

def get_classifier_input(tokenizer, batch, device, max_length):
    sentences1 = [batch[SENTENCE_PAIR][0][i] for i in range(len(batch[SENTENCE_PAIR][0]))]
    sentences2 = [batch[SENTENCE_PAIR][1][i] for i in range(len(batch[SENTENCE_PAIR][1]))]
    examples = [Example(sentence1, sentence2, label) for sentence1, sentence2, label in
                zip(sentences1, sentences2, batch[LABEL])]
    features = glue_convert_examples_to_features(examples, tokenizer=tokenizer, label_list=LABEL_LIST,
                                                 output_mode=CLASSIFICATION, max_length=max_length)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features])

    all_input_ids = all_input_ids.to(device)
    all_attention_mask = all_attention_mask.to(device)
    all_token_type_ids = all_token_type_ids.to(device)
    all_labels = all_labels.to(device)

    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def test(device, tokenizer, test_iterator, max_length, model):
    model.eval()
    total_corrects = 0
    total = 0

    for batch in tqdm(test_iterator, desc="Test batches"):
        input_ids, attention_mask, token_type_ids, labels = get_classifier_input(tokenizer, batch, device,
                                                                                 max_length)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        outputs = model(inputs)
        predictions = torch.argmax(outputs[0], dim = 1)
        corrects = (predictions == labels).long().sum()
        num_corrects = corrects.cpu().detach().numpy()
        total_corrects += num_corrects
        total += len(predictions)
    total_corrects = float(total_corrects)
    total = float(total)
    print(total_corrects, total)
    print("Accuracy : ", total_corrects/total)


def train_test(device, num_epochs, tokenizer, train_iterator, test_iterator, max_length):
    model = NLIClassifier(device)
    model = nn.DataParallel(model)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    steps = 0
    running_loss = 0
    for i in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        for batch in tqdm(train_iterator, desc="Batches"):
            input_ids, attention_mask, token_type_ids, labels = get_classifier_input(tokenizer, batch, device, max_length)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids" : token_type_ids}
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs[0], labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 10 == 0:
                print(running_loss/ 1000)
                running_loss = 0
                test(device, tokenizer, test_iterator, max_length, model)

def main():
    args = parse_args()
    train_set_nli = NLIDataset(args.train_file)
    test_set_nli = NLIDataset(args.test_file)
    train_iterator = DataLoader(train_set_nli, batch_size=args.batch_size)
    test_iterator = DataLoader(test_set_nli, shuffle=False, batch_size=args.batch_size)
    device = torch.device("cuda:0")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_test(device, args.num_epochs, tokenizer, train_iterator, test_iterator, args.max_length)


if __name__ == "__main__":
    main()
