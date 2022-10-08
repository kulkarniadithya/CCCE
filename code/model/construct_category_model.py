from transformers import BertModel
import torch.nn as nn


class CategoryModel(nn.Module):

    def __init__(self, output_labels):
        super(CategoryModel, self).__init__()
        self.module = BertModel.from_pretrained('bert-base-chinese')
        D_in, H1, D_out = 768, 256, output_labels
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.ReLU(),
            nn.Linear(H1, D_out)
        )

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        outputs = self.module(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        model_logits = self.classifier(last_hidden_state_cls)
        return model_logits, outputs
    