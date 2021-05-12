from modeling import BertForPreTraining


class BertForPreTraining_deepspeed(BertForPreTraining):
    def __init__(self, config, criterion):
        super(BertForPreTraining_deepspeed, self).__init__(config)

        self.criterion = criterion

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_labels):
        prediction_scores, seq_relationship_score = super().forward(input_ids, token_type_ids, attention_mask)

        loss = self.criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)

        return loss
