import pickle
from collections import Counter
from nltk import word_tokenize
from nltk.parse.corenlp import CoreNLPParser


def tokenize_words_chinese(text, parser):
    split = list(parser.tokenize(text))
    output = []
    for i in range(0, len(split)):
        if '-' not in split[i]:
            output.append(split[i])
        else:
            split1 = split[i].split("-")
            for j in range(0, len(split1)):
                if j%2==0:
                    output.append(split1[j])
                else:
                    output.append('-')
                    output.append(split1[j])
    return output


def tokenize_words(text):
    split = word_tokenize(text)
    output = []
    for i in range(0, len(split)):
        if '-' not in split[i]:
            output.append(split[i])
        else:
            split1 = split[i].split("-")
            for j in range(0, len(split1)):
                if j%2==0:
                    output.append(split1[j])
                else:
                    output.append('-')
                    output.append(split1[j])
    return output


def get_clause_index(text, word_to_index):
    words = tokenize_words(text)
    indexes = []
    for i in range(0, len(words)):
        try:
            index = word_to_index[words[i]]
            for j in range(0, len(index)):
                indexes.append(index[j])
        except:
            xyz = 1
    indexes.sort()
    final_indexes = get_index_sequence(indexes)
    return final_indexes


def get_index_sequence(indexes):
    updated_indexes = []
    for i in range(0, len(indexes)):
        if indexes[i] not in updated_indexes:
            updated_indexes.append(indexes[i])
    final_indexes = []
    if len(updated_indexes) > 1:
        for i in range(0, len(updated_indexes)):
            if i == 0:
                diff = updated_indexes[i + 1] - updated_indexes[i]
                if diff < 2:
                    final_indexes.append(updated_indexes[i])
            elif (i > 0) and (i < len(updated_indexes) - 1):
                diff1 = updated_indexes[i + 1] - updated_indexes[i]
                diff2 = updated_indexes[i] - updated_indexes[i - 1]
                if diff1 < 2 or diff2 < 2:
                    final_indexes.append(updated_indexes[i])
            else:
                diff2 = updated_indexes[i] - updated_indexes[i - 1]
                if diff2 < 2:
                    final_indexes.append(updated_indexes[i])
        return final_indexes
    else:
        return updated_indexes


def get_ground_truth_emotion_cause_pair(instance):
    clause = instance['clause']
    clause_keys = list(clause.keys())
    cause_index = []
    emotion_index = []
    emotion = 'null'
    for i in range(0, len(clause_keys)):
        if clause[clause_keys[i]]['cause'] != 'null':
            cause_index.append(clause_keys[i])
        if clause[clause_keys[i]]['emotion'] != 'null':
            emotion_index.append(clause_keys[i])
            emotion = clause[clause_keys[i]]['emotion']
    emotion_index_pair = []
    if len(emotion_index) > 0:
        for i in range(0, len(cause_index)):
            temp = [cause_index[i], emotion_index[0]]
            emotion_index_pair.append(temp)
    else:
        emotion = [" ".join(word_tokenize(clause[clause_keys[0]]['text'])[:3])]
    return cause_index, emotion_index, emotion_index_pair, emotion


def get_indexes(address_list, word_list):
    index_to_word = {}
    for i in range(0, len(address_list)):
        index_to_word[address_list[i]] = word_list[i]
    counter = Counter(word_list)
    keys = list(Counter(word_list).keys())
    word_to_index = {}
    for i in range(0, len(keys)):
        temp = []
        if counter[keys[i]] > 1:
            for j in range(0, len(index_to_word)):
                if index_to_word[j] == keys[i]:
                    temp.append(j)
        else:
            for j in range(0, len(index_to_word)):
                if index_to_word[j] == keys[i]:
                    temp.append(j)
                    break

        word_to_index[keys[i]] = temp
    return index_to_word, word_to_index


def get_emotion_index_sequence(emotion, word_to_index, clause_index):
    words = tokenize_words(emotion)
    temp = []
    for j in range(0, len(words)):
        try:
            word_index = word_to_index[words[j]]
            if len(word_index) == 1:
                temp.append(word_index[0])
            else:
                temp1 = []
                for k in range(0, len(word_index)):
                    if word_index[k] in clause_index:
                        temp1.append(word_index[k])
                for z in range(0, len(temp1)):
                    temp.append(temp1[z])
        except:
            print("Error")
            print(words[j])
            print(word_to_index)
    if len(words) > 2:
        temp = get_index_sequence(temp)
    return temp


def get_word_word_dictionary(address_list, head_list):
    word_word_dictionary = {}
    for i in range(0, len(address_list)):
        word_word_dictionary[address_list[i]] = []
    for i in range(0, len(head_list)):
        if head_list[i] != -1:
            word_word_dictionary[head_list[i]].append(address_list[i])
    return word_word_dictionary


if __name__ == "__main__":
    train_write_path = "../../processed_data/Chinese/data_dictionary_train.pickle"
    with open(train_write_path, 'rb') as handle:
        read_train_data = pickle.load(handle)

    parser = CoreNLPParser('http://localhost:9005')
    indexed_data = {}
    for a in range(0, len(read_train_data)):
        address_list = read_train_data[a]['dependency']['address_list']
        head_list = read_train_data[a]['dependency']['head_list']
        word_list = read_train_data[a]['dependency']['word_list']
        word_word_dictionary = get_word_word_dictionary(address_list, head_list)
        index_to_word, word_to_index = get_indexes(address_list, word_list)
        cause_index, emotion_index, emotion_index_pair, emotion = get_ground_truth_emotion_cause_pair(read_train_data[a])
        clause = read_train_data[a]['clause']
        clause_keys = list(clause.keys())
        indexed_clause = {}
        for i in range(0, len(clause_keys)):
            clause_index_sequence = get_clause_index(clause[clause_keys[i]]['text'], word_to_index)
            clause_emotion = clause[clause_keys[i]]['emotion']
            clause_cause = clause[clause_keys[i]]['cause']
            clause_emotion_index = []
            if clause_emotion != 'null':
                for y in range(0, len(clause_emotion)):
                    clause_emotion_index.append(get_emotion_index_sequence(clause_emotion[y], word_to_index, clause_index_sequence))
            clause_cause_index = []
            if clause_cause != 'null':
                for y in range(0, len(clause_cause)):
                    clause_cause_index.append(get_emotion_index_sequence(clause_cause[y], word_to_index, clause_index_sequence))
            indexed_clause[clause_keys[i]] = {}
            indexed_clause[clause_keys[i]]['index_sequence'] = clause_index_sequence
            indexed_clause[clause_keys[i]]['emotion_index'] = clause_emotion_index
            indexed_clause[clause_keys[i]]['cause_index'] = clause_cause_index

        indexed_data[a] = {}
        indexed_data[a]['category'] = read_train_data[a]['category']
        indexed_data[a]['word_word_dictionary'] = word_word_dictionary
        indexed_data[a]['index_to_word'] = index_to_word
        indexed_data[a]['word_to_index'] = word_to_index
        indexed_data[a]['cause_index'] = cause_index
        indexed_data[a]['emotion_index'] = emotion_index
        indexed_data[a]['emotion_index_pair'] = emotion_index_pair
        if len(emotion_index) > 0:
            indexed_data[a]['emotion'] = [emotion, get_emotion_index_sequence(emotion[0], word_to_index, indexed_clause[emotion_index[0]]['index_sequence'])]
        else:
            indexed_data[a]['emotion'] = [emotion, [0, 1, 2, 3]]

        indexed_data[a]['indexed_clause'] = indexed_clause
        indexed_data[a]['dependency'] = read_train_data[a]['dependency']

    print(indexed_data[10])

    train_write_path_index = "../../processed_data/Chinese/data_dictionary_train_updated_indexed.pickle"

    with open(train_write_path_index, 'wb') as handle:
        pickle.dump(indexed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
