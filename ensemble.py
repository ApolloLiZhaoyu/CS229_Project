if __name__ == '__main__':
    import numpy as np
    ensemble_set = [1, 2, 3, 4, 5, 6, 7]
    num_ensemble = len(ensemble_set)
    num_result = 1063
    result = np.zeros(num_result)
    prob1s = np.zeros(num_result)
    prob2s = np.zeros(num_result)
    for i in ensemble_set:
        with open('CoLA_{}.tsv'.format(i), 'r') as f:
            print(i)
            for idx, line in enumerate(f.readlines()):
                data = line.split()
                prob = data[1]
                result[idx] += float(prob)
                # prob1, prob2 = data[2], data[3]
                # prob1s[idx] += float(prob1)
                # prob2s[idx] += float(prob2)

    # for idx in range(num_result):
    #     result[idx] = np.array([prob1s[idx], prob2s[idx]]).argmax()

    result /= num_ensemble
    result[result >= 0.5] = 1
    result[result < 0.5] = 0

    with open('CoLA.tsv', 'w') as f:
        f.write('index\tprediction\n')
        for idx in range(num_result):
            f.write('{}\t{}\n'.format(idx, int(result[idx])))



