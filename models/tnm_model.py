from transformers import AutoModel
from torch import nn
import numpy as np


# Multitask BERT classification model with four heads
class TNMModel(nn.Module):

    def __init__(self, args):

        super(TNMModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_ckpt, return_dict=False)
        self.drop = nn.Dropout(args.dropout_prob)

        # Tumour head
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1)

        # Node head
        self.fc2 = nn.Linear(self.bert.config.hidden_size, 1)

        # Mets head
        self.fc3 = nn.Linear(self.bert.config.hidden_size, 1)

        # Uncertainty head
        self.fc4 = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids,
                                     attention_mask=attention_mask)
        dropoutput = self.drop(pooled_output)

        output1 = self.fc1(dropoutput)
        output2 = self.fc2(dropoutput)
        output3 = self.fc3(dropoutput)
        output4 = self.fc4(dropoutput)

        return {"u": output1, "t": output2, "n": output3, "m":output4}

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
