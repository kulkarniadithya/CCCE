import pickle
import xml.etree.ElementTree as Et


def format_data(file_path):
    tree = Et.parse(file_path)
    root = tree.getroot()
    sentence_dictionary = {}
    for i in range(0, len(root)):
        id = int(root[i].attrib['id'])
        sentence_dictionary[id] = {}
        sentence_dictionary[id]['category'] = {}
        sentence_dictionary[id]['category']['name'] = root[i][0].attrib['name']
        sentence_dictionary[id]['category']['value'] = root[i][0].attrib['value']
        sentence_dictionary[id]['clause'] = {}
        for j in range(1, len(root[i])):
            cause_value = root[i][j].attrib['cause']
            keywords_value = root[i][j].attrib['keywords']
            clause_id = root[i][j].attrib['id']
            sentence_dictionary[id]['clause'][clause_id] = {}
            if cause_value == 'N' and keywords_value == 'N':
                try:
                    sentence_dictionary[id]['clause'][clause_id]['text'] = root[i][j][0].text
                    sentence_dictionary[id]['clause'][clause_id]['cause'] = 'null'
                    sentence_dictionary[id]['clause'][clause_id]['emotion'] = 'null'
                except:
                    sentence_dictionary[id]['clause'][clause_id]['text'] = 'null'
                    sentence_dictionary[id]['clause'][clause_id]['cause'] = 'null'
                    sentence_dictionary[id]['clause'][clause_id]['emotion'] = 'null'
            elif cause_value == 'Y' and keywords_value == 'N':
                sentence_dictionary[id]['clause'][clause_id]['text'] = root[i][j][0].text
                cause_array = []
                for k in range(1, len(root[i][j])):
                    try:
                        cause_array.append(root[i][j][k].text)
                    except:
                        print("Error")
                sentence_dictionary[id]['clause'][clause_id]['cause'] = cause_array
                sentence_dictionary[id]['clause'][clause_id]['emotion'] = 'null'
            elif cause_value == 'Y' and keywords_value == 'Y':
                sentence_dictionary[id]['clause'][clause_id]['text'] = root[i][j][0].text
                cause_array = []
                emotion_array = []
                for k in range(1, len(root[i][j])):
                    try:
                        cause_array.append(root[i][j][k].text)
                    except:
                        emotion_array.append(root[i][j][k].text)
                sentence_dictionary[id]['clause'][clause_id]['cause'] = cause_array
                sentence_dictionary[id]['clause'][clause_id]['emotion'] = emotion_array
            elif cause_value == 'N' and keywords_value == 'Y':
                sentence_dictionary[id]['clause'][clause_id]['text'] = root[i][j][0].text
                emotion_array = []
                for k in range(1, len(root[i][j])):
                    try:
                        emotion_array.append(root[i][j][k].text)
                    except:
                        print("Error")
                sentence_dictionary[id]['clause'][clause_id]['cause'] = 'null'
                sentence_dictionary[id]['clause'][clause_id]['emotion'] = emotion_array
    return sentence_dictionary


if __name__ == "__main__":
    read_path = "../../original_data/emotion_cause_chi_train.xml"
    write_path = "../../processed_data/Chinese/data_dictionary_train.pickle"

    formatted_sentence_dictionary = format_data(file_path=read_path)
    print(len(formatted_sentence_dictionary))
    with open(write_path, 'wb') as handle:
        pickle.dump(formatted_sentence_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(write_path, 'rb') as handle:
        read_data = pickle.load(handle)
    assert formatted_sentence_dictionary == read_data
    print(len(read_data))
    print(read_data[0])
