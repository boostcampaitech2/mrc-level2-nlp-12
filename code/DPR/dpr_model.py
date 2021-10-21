from transformers import (
    TrainingArguments,
    BertPreTrainedModel,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel,
    AutoConfig,
)
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, model_checkpoint):
        super().__init__()
        self.model_checkpoint = model_checkpoint
        config = AutoConfig.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint, config=config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        return pooled_output


class BertEncoder(BertPreTrainedModel):
    """A class for encoding questions and passages
    """

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        # embedded vec
        pooled_output = outputs[1]
        return pooled_output
