import pickle

f = open('glue_data/models/electra_large/test_predictions/cola_test_1_predictions.pkl','rb')
data = pickle.load(f)

with open('result.tsv', 'w+') as f:
    for idx in data:
        f.write('{}\t{}\t{}\t{}\n'.format(idx, data[idx].argmax(), data[idx][0], data[idx][1]))