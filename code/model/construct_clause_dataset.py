import torch
from torch.utils.data import Dataset


def get_formatted_index(emotion_index):
    final_indexes = []
    if len(emotion_index) > 1:
        for i in range(0, len(emotion_index)):
            if i == 0:
                diff = emotion_index[i + 1] - emotion_index[i]
                if diff < 2:
                    final_indexes.append(emotion_index[i])
            elif (i > 0) and (i < len(emotion_index) - 1):
                diff1 = emotion_index[i + 1] - emotion_index[i]
                diff2 = emotion_index[i] - emotion_index[i - 1]
                if diff1 < 2 and diff2 < 2:
                    final_indexes.append(emotion_index[i])
                elif diff2 < 2:
                    final_indexes.append(emotion_index[i])
            else:
                diff2 = emotion_index[i] - emotion_index[i - 1]
                if diff2 < 2:
                    final_indexes.append(emotion_index[i])
        return final_indexes
    else:
        return emotion_index


class ConstructDatasetSetting2(Dataset):
    def __init__(self, dataset_review, dataset_emotion, dataset_clause_text, dataset_label, tokenizer):
        self.dataset_review = dataset_review
        self.dataset_emotion = dataset_emotion
        self.dataset_clause_text = dataset_clause_text
        self.dataset_label = dataset_label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_review)

    def __getitem__(self, idx):
        review_text = self.dataset_review[idx].strip()
        emotion_text = self.dataset_emotion[idx].strip()
        clause_text = self.dataset_clause_text[idx].strip()
        instance_label = self.dataset_label[idx]
        if str(emotion_text) == "":
            model_input = "<衣> </衣> " + str(review_text) + " [SEP] " + str(clause_text)
        else:
            model_input = "<衣> " + str(emotion_text) + " </衣> " + str(review_text) + " [SEP] " + str(clause_text)

        encoding = self.tokenizer.encode_plus(model_input, max_length=256,
                                              add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                              return_token_type_ids=True, padding='max_length',
                                              return_attention_mask=True,
                                              return_tensors='pt', truncation=True)

        inputs = model_input
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        token_type_ids = encoding['token_type_ids'].flatten()
        labels = torch.tensor(instance_label, dtype=torch.long)

        sample = {"inputs": inputs, "input_ids": input_ids, "attention_mask": attention_mask,
                  "token_type_ids": token_type_ids, "labels": labels}
        return sample
