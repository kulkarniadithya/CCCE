import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForTokenClassification, BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from collections import defaultdict
from construct_clause_dataset import ConstructDatasetSetting2
from construct_clause_model import Model
from clause_evaluation import get_report
from sklearn.cross_decomposition import CCA
from construct_category_dataset import ConstructDatasetCategorySetting2
from construct_category_model import CategoryModel
from category_evaluation import get_category_report


def process_dataset(dictionary):
    dataset_review = []
    dataset_clause_text = []
    dataset_emotion = []
    dataset_label = []

    dataset_review_suffix = []
    dataset_clause_text_suffix = []
    dataset_emotion_suffix = []
    dataset_label_suffix = []

    for i in range(0, len(dictionary)):
        clause = dictionary[i]['clause']
        clause_keys = list(clause.keys())
        temp_clause_text = []
        temp_emotion = []
        temp_label = []
        for j in range(0, len(clause_keys)):
            if clause[clause_keys[j]]['cause'] != 'null':
                if clause[clause_keys[j]]['emotion'] != 'null':
                    emotion_text = clause[clause_keys[j]]['emotion'][0].strip()
                    instance_clause_text = str(clause[clause_keys[j]]['text']).strip()
                    temp_label.append(1)
                    temp_emotion.append(emotion_text)
                    temp_clause_text.append(instance_clause_text)
                else:
                    instance_clause_text = str(clause[clause_keys[j]]['text']).strip()
                    temp_label.append(1)
                    temp_clause_text.append(instance_clause_text)
            else:
                if clause[clause_keys[j]]['emotion'] != 'null':
                    emotion_text = clause[clause_keys[j]]['emotion'][0].strip()
                    instance_clause_text = str(clause[clause_keys[j]]['text']).strip()
                    temp_label.append(0)
                    temp_emotion.append(emotion_text)
                    temp_clause_text.append(instance_clause_text)
                else:
                    instance_clause_text = str(clause[clause_keys[j]]['text']).strip()
                    temp_label.append(0)
                    temp_clause_text.append(instance_clause_text)

        review_text = " ".join(temp_clause_text)
        assert len(temp_clause_text) == len(temp_label)
        for k in range(0, len(temp_clause_text)):
            if k == 0:
                dataset_review.append(review_text)
                if len(temp_emotion) > 0:
                    dataset_emotion.append(temp_emotion[0])
                else:
                    dataset_emotion.append("")
                dataset_clause_text.append(temp_clause_text[k])
                dataset_label.append(temp_label[k])
            else:
                dataset_review_suffix.append(review_text)
                if len(temp_emotion) > 0:
                    dataset_emotion_suffix.append(temp_emotion[0])
                else:
                    dataset_emotion_suffix.append("")
                dataset_clause_text_suffix.append(temp_clause_text[k])
                dataset_label_suffix.append(temp_label[k])

    for x in range(0, len(dataset_review_suffix)):
        dataset_review.append(dataset_review_suffix[x])
        dataset_emotion.append(dataset_emotion_suffix[x])
        dataset_clause_text.append(dataset_clause_text_suffix[x])
        dataset_label.append(dataset_label_suffix[x])

    return dataset_review, dataset_emotion, dataset_clause_text, dataset_label


def model_training(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train().to(device)
    losses = []
    correct_predictions = 0
    epoch_bert_output = []
    for i, batch in enumerate(tqdm(data_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch["labels"].to(device)

        outputs, bert_output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        bert_input_ids = np.array(input_ids.squeeze().detach().cpu().tolist())
        bert_embed = bert_output[0].squeeze().detach().cpu()
        for x in range(0, len(bert_input_ids)):
            start_index = np.where(bert_input_ids[x] == 101)[0][0]
            end_index = np.where(bert_input_ids[x] == 102)[0][0]
            epoch_bert_output.append(bert_embed[x][start_index + 1:end_index])
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses), epoch_bert_output


def model_training_category(category_model, train_data_category, loss_fn_category, category_optimizer, device,
                            scheduler_category, n_examples):
    category_model.train().to(device)
    losses = []
    correct_predictions = 0
    epoch_bert_output = []
    for i, batch in enumerate(tqdm(train_data_category)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch["labels"].to(device)

        outputs, bert_output = category_model(input_ids=input_ids, token_type_ids=token_type_ids,
                                              attention_mask=attention_mask)
        bert_input_ids = np.array(input_ids.squeeze().detach().cpu().tolist())
        bert_embed = bert_output[0].squeeze().detach().cpu()
        for x in range(0, len(bert_input_ids)):
            start_index = np.where(bert_input_ids[x] == 101)[0][0]
            end_index = np.where(bert_input_ids[x] == 102)[0][0]
            epoch_bert_output.append(bert_embed[x][start_index + 1:end_index])
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn_category(outputs, labels)
        correct_predictions += torch.sum(preds == labels)

        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(category_model.parameters(), max_norm=1.0)

        category_optimizer.step()
        scheduler_category.step()
        category_optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses), epoch_bert_output


if __name__ == "__main__":
    train_read_path = "../../processed_data/Chinese/data_dictionary_train.pickle"
    with open(train_read_path, 'rb') as handle:
        train = pickle.load(handle)

    test_read_path = "../../processed_data/Chinese/data_dictionary_test.pickle"
    with open(test_read_path, 'rb') as handle:
        test = pickle.load(handle)

    print(train[0])
    print(test[0])

    train_dataset_review, train_dataset_emotion, train_dataset_clause_text, train_dataset_label = process_dataset(train)
    test_dataset_review, test_dataset_emotion, test_dataset_clause_text, test_dataset_label = process_dataset(test)

    PRE_TRAINED_MODEL_NAME = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    train_dataset = ConstructDatasetSetting2(dataset_review=train_dataset_review, dataset_emotion=train_dataset_emotion,
                                             dataset_clause_text=train_dataset_clause_text,
                                             dataset_label=train_dataset_label, tokenizer=tokenizer)
    test_dataset = ConstructDatasetSetting2(dataset_review=test_dataset_review, dataset_emotion=test_dataset_emotion,
                                            dataset_clause_text=test_dataset_clause_text,
                                            dataset_label=test_dataset_label, tokenizer=tokenizer)

    emotion_mapping = {"surprise": 0, "disgust": 1, "fear": 2, "happiness": 3, "anger": 4, "sadness": 5}
    train_dataset_category = ConstructDatasetCategorySetting2(dictionary=train, emotion_mapping=emotion_mapping,
                                                              tokenizer=tokenizer)
    test_dataset_category = ConstructDatasetCategorySetting2(dictionary=test, emotion_mapping=emotion_mapping,
                                                             tokenizer=tokenizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(output_labels=2)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-5)

    category_model = CategoryModel(output_labels=6)
    category_optimizer = optim.AdamW(params=category_model.parameters(), lr=1e-5)

    n_epochs = 7
    train_data = DataLoader(train_dataset, batch_size=8)
    test_data = DataLoader(test_dataset, batch_size=8)
    total_steps = len(train_data) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2,
                                                num_training_steps=total_steps)

    loss_fn = nn.CrossEntropyLoss().to(device)

    train_data_category = DataLoader(train_dataset_category, batch_size=8)
    test_data_category = DataLoader(test_dataset_category, batch_size=8)
    total_steps_category = len(train_data_category) * n_epochs
    scheduler_category = get_linear_schedule_with_warmup(category_optimizer, num_warmup_steps=2,
                                                         num_training_steps=total_steps_category)

    loss_fn_category = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    history_category = defaultdict(list)
    best_accuracy = 0
    correlation_result = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('*' * 41)

        train_acc, train_loss, clause_bert_output = model_training(model, train_data, loss_fn, optimizer, device,
                                                                   scheduler,
                                                                   len(train_dataset_review))
        train_acc_category, train_loss_category, category_bert_output = model_training_category(category_model,
                                                                                                train_data_category,
                                                                                                loss_fn_category,
                                                                                                category_optimizer,
                                                                                                device,
                                                                                                scheduler_category,
                                                                                                len(train))

        temp_correlation_result = []
        for z in range(0, len(category_bert_output)):
            try:
                X = category_bert_output[z]
                Y = clause_bert_output[z][:len(X)]
                cca = CCA(n_components=4)
                cca.fit(X, Y)
                X_c, Y_c = cca.transform(X, Y)
                result = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                temp_correlation_result.append(result)
            except:
                print("Error")
        mean_correlation = np.mean(np.array(temp_correlation_result))
        correlation_result.append(mean_correlation)

        print(f'Clause Train loss {train_loss} accuracy {train_acc}')
        print(f'Category Train loss {train_loss_category} accuracy {train_acc_category}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        history_category['train_acc'].append(train_acc_category)
        history_category['train_loss'].append(train_loss_category)
        print(mean_correlation)

        torch.save(model, '../../saved_models/cca/clause_experiment_epoch_' + str(epoch))
        torch.save(category_model, '../../saved_models/cca/category_experiment_epoch_' + str(epoch))
    print("Correlation result")
    print(correlation_result)
    file_open = open("../../saved_models/cca/correlation_score_experiment.txt", "w")
    for x in range(0, len(correlation_result)):
        file_open.write(str(correlation_result[x]) + "\n")
    file_open.close()
