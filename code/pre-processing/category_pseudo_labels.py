import pickle
from transformers import BertModel, BertTokenizer
import torch
from numpy import dot
from numpy.linalg import norm


def calculate_cosine_similarity(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def get_category_label_updated(instance, category_labels, category_embedding, hs):
    clause = instance['clause']
    clause_keys = list(clause.keys())
    sentence = ""
    for i in range(0, len(clause_keys)):
        sentence = sentence + clause[clause_keys[i]]['text'] + " "
    emotion = [""]
    for i in range(0, len(clause_keys)):
        if clause[clause_keys[i]]['emotion'] != 'null':
            emotion = clause[clause_keys[i]]['emotion']
            break
    model_input = "[CLS] " + "<衣> " + str(emotion[0]) + " </衣> " + sentence + "[SEP]"
    tokenized_text = tokenizer.tokenize(model_input)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
    index_compare = []
    for i in range(0, len(indexed_tokens)):
        if indexed_tokens[i] in [101]:
            index_compare.append(i)
    embedding = hidden_states[hs][0]
    embedding_to_compare = []
    for i in range(0, len(index_compare)):
        embedding_to_compare.append(embedding[index_compare[i]].tolist())
    text_embedding = embedding_to_compare[0]
    cosine_sim = []
    for i in range(0, len(category_labels)):
        cosine_sim.append(calculate_cosine_similarity(text_embedding, category_embedding[i]))
    temp = cosine_sim.copy()
    temp.sort(reverse=True)
    label = category_labels[cosine_sim.index(temp[0])]
    return label, cosine_sim


def get_category_embedding(instance, category_labels, tokenizer, model, hs):
    clause = instance['clause']
    clause_keys = list(clause.keys())
    sentence = ""
    for i in range(0, len(clause_keys)):
        sentence = sentence + clause[clause_keys[i]]['text'] + " "
    category_embedding = []
    for j in range(0, len(category_labels)):
        model_input = "[CLS] " + "<衣> " + str(category_labels[j]) + " </衣> " + sentence + " [SEP]"
        tokenized_text = tokenizer.tokenize(model_input)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        index_compare = []
        for i in range(0, len(indexed_tokens)):
            if indexed_tokens[i] in [101]:
                index_compare.append(i)
        embedding = hidden_states[hs][0]
        embedding_to_compare = []
        for i in range(0, len(index_compare)):
            embedding_to_compare.append(embedding[index_compare[i]].tolist())
        category_embedding.append(embedding_to_compare[0])
    return category_embedding


if __name__ == "__main__":
    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)
    model.eval()
    for z in range(1, 13):
        train_write_path = "../../processed_data/Chinese/data_dictionary_train_updated.pickle"
        with open(train_write_path, 'rb') as handle:
            read_train_data = pickle.load(handle)
        print(read_train_data[0])
        category_labels = []
        for i in range(0, len(read_train_data)):
            category_name = read_train_data[i]['category']['name']
            if category_name not in category_labels:
                category_labels.append(category_name)
        print(category_labels)

        for i in range(0, len(read_train_data)):
            category_embedding = get_category_embedding(read_train_data[i], category_labels, tokenizer, model, z)
            pred_label, cosine_sim = get_category_label_updated(read_train_data[i], category_labels, category_embedding, z)
            read_train_data[i]['category_prediction'] = {}
            read_train_data[i]['category_prediction']['pred_category'] = pred_label
            read_train_data[i]['category_prediction']['cosine_sim'] = cosine_sim

        pseudo_write_path = "../../processed_data/Chinese/data_dictionary_train_pseudo_label_category_" + str(z) + ".pickle"

        with open(pseudo_write_path, 'wb') as handle:
            pickle.dump(read_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
