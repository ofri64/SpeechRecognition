from read_write_data import load_train_data, load_test_data, write_results
from models import predict_one_nn, DTW
# from omer_models import dtw_fast
import numpy as np

if __name__ == '__main__':

    train_data = load_train_data()
    test_data = load_test_data()

    euc_predictions = []
    dtw_predictions = []
    for _, test_sample in test_data:

        # euclidian distance 1-nn classifier
        euc_label = predict_one_nn(train_data, test_sample, lambda x, y: np.average(np.linalg.norm(x-y, axis=0)))
        euc_predictions.append(euc_label)

        # DTW distance 1-nn classifier
        dtw_label = predict_one_nn(train_data, test_sample, DTW)
        dtw_predictions.append(dtw_label)

    test_names = [name for name, sample in test_data]
    write_results(test_names, euc_predictions, dtw_predictions)
