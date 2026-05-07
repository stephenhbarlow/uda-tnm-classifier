from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training, PeftType
from peft import PeftModel, PeftConfig
from torch import nn
import numpy as np

# Multitask BERT classification model with four heads
class PeftModel(nn.Module):

    def __init__(self, model, args):

        super(PeftModel, self).__init__()
        self.model = model
        self.drop = nn.Dropout(args.dropout_prob)
        self.peft_type = PeftType.LORA

        # Classification head
        self.fc = nn.Linear(self.model.config.hidden_size, args.n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
        dropoutput = self.drop(pooled_output)

        output = self.fc(dropoutput)


        return output

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)