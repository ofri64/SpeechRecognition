import os
import librosa

LABEL_MAPPING = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def load_train_data():
    train_data = []
    train_folder, labels_folder, _ = next(os.walk("train_data"))
    for label_dir in labels_folder:
        abs_path = os.path.join(train_folder, label_dir)
        label = LABEL_MAPPING[label_dir]

        for file_name in os.listdir(abs_path):
            if not file_name.endswith(".wav"):
                continue

            # load frequency features from wav file using librosa
            file_path = os.path.join(abs_path, file_name)
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            train_data.append((mfcc, label))

    return train_data


def load_test_data():
    test_data = []
    test_folder = "test_files"
    for file_name in os.listdir(test_folder):

        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(test_folder, file_name)

        # load frequency features from wav file using librosa
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        test_data.append((file_name, mfcc))

    return test_data


def write_results(test_names, euc_predictions, dtw_predictions):
    with open("output.txt", "w") as f:
        test_sample_length = len(test_names)
        for i in range(len(test_names)):
            current_test = test_names[i]
            euc_pred = euc_predictions[i]
            dtw_pred = dtw_predictions[i]
            row_data = "{0} - {1} - {2}".format(current_test, euc_pred, dtw_pred)

            if i < test_sample_length-1:
                row_data += "\n"

            f.write(row_data)
