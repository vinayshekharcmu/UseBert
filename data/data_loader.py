from torch.utils.data import Dataset
import jsonlines
import json
from config import *
from transformers import glue_convert_examples_to_features as convert_examples_to_features

def read_file(file_name):
    examples = []
    with open(file_name) as reader:
        for obj in reader:
            obj = json.loads(obj)
            sentence1 = obj[SENTENCE1]
            sentence2 = obj[SENTENCE2]
            gold_label = obj[GOLD_LABEL]
            examples.append([sentence1, sentence2, gold_label])
    return examples



class NLIDataset(Dataset):

    def __init__(self, file_name):
        self.examples = read_file(file_name)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sentence1 = self.examples[index][0]
        sentence2 = self.examples[index][1]
        gold_label = self.examples[index][2]

        return {SENTENCE_PAIR : [sentence1, sentence2], LABEL : gold_label}

if __name__ == "__main__":
    read_file("../../Data/snli_1.0/snli_1.0_test.jsonl")

