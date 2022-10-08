from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
import pickle


def get_dependency(parser, text):
    dep = next(parser.raw_parse(text))
    address_list = []
    head_list = []
    relation_list = []
    tag_list = []
    word_list = []
    for i in range(1, len(dep.nodes)):
        address_list.append(int(dep.nodes[i]['address']) - 1)
        head_list.append(int(dep.nodes[i]['head']) - 1)
        relation_list.append(str(dep.nodes[i]['rel']).split(":")[0])
        tag_list.append(str(dep.nodes[i]['tag']))
        word_list.append(str(dep.nodes[i]['word']))
    return address_list, head_list, relation_list, tag_list, word_list


def find_relation(dictionary, dep_parser):
    for i in range(0, len(dictionary)):
        clause_text_list = []
        clause = dictionary[i]['clause']
        keys = list(clause.keys())
        print(keys)
        for j in range(0, len(keys)):
            text = clause[keys[j]]['text']
            if text != 'null':
                clause_text_list.append(text)
        sentence = " ".join(clause_text_list)
        address_list, head_list, relation_list, tag_list, word_list = get_dependency(dep_parser, sentence)
        dictionary[i]['dependency'] = {}
        dictionary[i]['dependency']['address_list'] = address_list
        dictionary[i]['dependency']['head_list'] = head_list
        dictionary[i]['dependency']['relation_list'] = relation_list
        dictionary[i]['dependency']['tag_list'] = tag_list
        dictionary[i]['dependency']['word_list'] = word_list
    return dictionary


if __name__ == "__main__":
    train_read_path = "../../processed_data/Chinese/data_dictionary_train.pickle"
    with open(train_read_path, 'rb') as handle:
        train = pickle.load(handle)
    print(train[0])
    print(train[0])
    dep_parser = CoreNLPDependencyParser('http://localhost:9005')
    updated_train = find_relation(train, dep_parser)

    print(updated_train[0])

    train_write_path = "../../processed_data/Chinese/data_dictionary_train_updated.pickle"

    with open(train_write_path, 'wb') as handle:
        pickle.dump(updated_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(train_write_path, 'rb') as handle:
        read_train_data = pickle.load(handle)
    assert updated_train == read_train_data
    print(len(read_train_data))
    print(len(updated_train))
