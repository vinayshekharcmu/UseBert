import torch
from transformers import PreTrainedTokenizer, BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader
from data.data_loader import LineByLineDataset
import torch.nn as nn
import argparse
from config import *

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(model: BertForMaskedLM, tokenizer, train_iterator, device, steps):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters())
    running_loss = 0
    for batch in train_iterator:
        sentences = batch[SENTENCE]
        inputs = tokenizer.batch_encode_plus(sentences, pad_to_max_length = True, truncate = True)
        attention_mask = torch.tensor(inputs["attention_mask"])
        token_type_ids = torch.tensor(inputs["token_type_ids"])
        input_ids, labels = mask_tokens(torch.tensor(inputs["input_ids"]), tokenizer)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        predictions = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]
        predictions = predictions.view(-1, predictions.shape[2])
        labels = labels.view(-1)
        loss = loss_fn( predictions , labels)
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        steps += 1
        optimizer.step()

        if steps % 100 == 0:
            print(running_loss / 100)
            running_loss = 0



def eval(model, tokenizer, eval_iterator, device):
    model.eval()


def train_iters(num_epochs, model, tokenizer, train_iterator, eval_iterator, device):
    steps = 0
    for i in range(num_epochs):
        steps = train(model, tokenizer, train_iterator, device, steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file")
    parser.add_argument("--test_file")
    parser.add_argument("--num_epochs", default=3)
    parser.add_argument("--batch_size", default=16)
    args = parser.parse_args()

    train_dataset = LineByLineDataset(args.train_file)
    eval_dataset = LineByLineDataset(args.test_file)

    train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_iterator = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    device = torch.device("cuda:0")

    model.to(device)

    train_iters(args.num_epochs, model, tokenizer, train_iterator, eval_iterator, device)


