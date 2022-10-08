import pickle
import numpy as np


def get_pseudo_label(instance):
    emotion = instance['emotion'][1]
    dependency = instance['dependency']
    head_list = dependency['head_list']
    indexed_clause = instance['indexed_clause']
    word_word_dictionary = instance['word_word_dictionary']
    clause_keys = list(indexed_clause.keys())
    indexed_values = []
    for i in range(0, len(clause_keys)):
        indexed_values.append(indexed_clause[clause_keys[i]]['index_sequence'])
    emotion_head = []
    for i in range(0, len(emotion)):
        dep = word_word_dictionary[emotion[i]]
        for j in range(0, len(dep)):
            if dep[j] not in emotion_head:
                emotion_head.append(dep[j])

    incoming_emotion_head = []
    for i in range(0, len(emotion)):
        incoming_emotion_head.append(head_list[emotion[i]])

    incoming_percentage = [0] * len(clause_keys)
    for i in range(0, len(indexed_values)):
        match = 0
        for j in range(0, len(indexed_values[i])):
            if indexed_values[i][j] in incoming_emotion_head:
                match = match + 1
        if len(incoming_emotion_head) > 0:
            incoming_percentage[i] = match / len(incoming_emotion_head)

    percentage = [0] * len(clause_keys)
    for i in range(0, len(indexed_values)):
        match = 0
        for j in range(0, len(indexed_values[i])):
            if indexed_values[i][j] in emotion_head:
                match = match + 1
        if len(emotion_head) > 0:
            percentage[i] = match / len(emotion_head)
    temp = []
    for i in range(0, len(percentage)):
        temp.append((0.3*percentage[i] + 0.7*incoming_percentage[i])/2)
    aggregate = []
    for i in range(0, len(temp)):
        try:
            aggregate.append(round(temp[i]/np.sum(temp), 4))
        except:
            print(temp[i])
            print(temp)
    max_value = max(aggregate)
    pred_cause_index = []
    if max_value > 0:
        for i in range(0, len(aggregate)):
            if aggregate[i] > 0:
                pred_cause_index.append(clause_keys[i])
    return aggregate, pred_cause_index


if __name__ == "__main__":
    train_write_path = "../../processed_data/Chinese/data_dictionary_train_updated_indexed.pickle"
    with open(train_write_path, 'rb') as handle:
        read_train_data = pickle.load(handle)
    print(read_train_data[0])
    for i in range(0, len(read_train_data)):
        percentage, pred_cause_index = get_pseudo_label(read_train_data[i])
        read_train_data[i]['prediction'] = {}
        read_train_data[i]['prediction']['confidence_percentage'] = percentage
        read_train_data[i]['prediction']['pred_cause_index'] = pred_cause_index

    pseudo_write_path = "../../processed_data/Chinese/data_dictionary_train_pseudo_label_average.pickle"

    with open(pseudo_write_path, 'wb') as handle:
        pickle.dump(read_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
