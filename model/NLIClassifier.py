from transformers import BertForSequenceClassification, BertConfig
import torch.nn as nn

class NLIClassifier(nn.Module):

    def __init__(self, device):
        super(NLIClassifier, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased', num_labels=3)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = self.config)
        self.bert.num_labels = 3
        self.bert = self.bert.to(device)

    def forward(self, inputs):
        output = self.bert(**inputs)
        return output
