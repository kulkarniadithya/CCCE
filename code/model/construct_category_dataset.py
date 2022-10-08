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


class ConstructDatasetCategorySetting2(Dataset):
    def __init__(self, dictionary, emotion_mapping, tokenizer):
        self.dictionary = dictionary
        self.emotion_mapping = emotion_mapping
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        clause = self.dictionary[idx]['clause']
        category = self.dictionary[idx]['category']['name']
        clause_keys = list(clause.keys())
        review = ""
        emotion = ""
        boolean = False
        for i in range(0, len(clause_keys)):
            if clause[clause_keys[i]]['emotion'] != 'null':
                boolean = True
                emotion = clause[clause_keys[i]]['emotion'][0].strip()
                review = review + str(clause[clause_keys[i]]['text']).strip() + " "
            else:
                review = review + str(clause[clause_keys[i]]['text']).strip() + " "
        if boolean:
            review = "<衣> " + str(emotion) + " </衣> " + str(review).rstrip()
        else:
            review = "<衣> </衣> " + str(review).rstrip()

        encoding = self.tokenizer.encode_plus(review, max_length=256,
                                              add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                              return_token_type_ids=True, padding='max_length',
                                              return_attention_mask=True,
                                              return_tensors='pt', truncation=True)

        inputs = review
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        token_type_ids = encoding['token_type_ids'].flatten()
        labels = torch.tensor(self.emotion_mapping[category], dtype=torch.long)

        sample = {"inputs": inputs, "input_ids": input_ids, "attention_mask": attention_mask,
                  "token_type_ids": token_type_ids, "labels": labels}
        return sample