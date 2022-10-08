import torch
import numpy as np
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import pickle


def model_evaluation(model, data_loader, loss_fn, device, n_examples):
    model = model.eval().to(device)
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch["labels"].to(device)
            outputs, bert_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader, device):

    model = model.eval().to(device)

    input_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():

        for i, batch in enumerate(tqdm(data_loader)):
            inputs = batch["inputs"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch["labels"].to(device)

            outputs, bert_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            input_texts.extend(inputs)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return input_texts, predictions, prediction_probs, real_values


def evaluate_data(model, data_loader, loss_fn, device, n_examples):
    model = model.eval().to(device)
    losses = []
    correct_predictions = 0
    input_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            inputs = batch["inputs"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch["labels"].to(device)
            outputs, bert_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                          attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            input_texts.extend(inputs)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return input_texts, predictions, prediction_probs, real_values, correct_predictions.double() / n_examples, np.mean(losses)


def get_report(trained_model, test_data_loader, loss_fn, device, n_examples, epoch):
    y_input_texts, y_pred, y_pred_probs, y_test, test_acc, loss = evaluate_data(trained_model, test_data_loader, loss_fn, device, n_examples)
    dictionary = {}
    for i in range(0, len(y_input_texts)):
        dictionary[i] = {}
        dictionary[i]['y_input_texts'] = y_input_texts[i]
        dictionary[i]['y_pred'] = y_pred[i]
        dictionary[i]['y_pred_probs'] = y_pred_probs[i]
        dictionary[i]['y_test'] = y_test[i]
    write_path = "../../Results/Prediction/cca/clause_experiment_chinese_" + str(epoch) + ".pickle"
    with open(write_path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Test loss : ", loss.item())
    print("Test accuracy : ", test_acc.item())
    print(classification_report(y_test, y_pred))
