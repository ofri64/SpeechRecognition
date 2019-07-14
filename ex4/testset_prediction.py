import torch


from gcommand_loader import DatasetLoaderParser, TestDataLoaderParser
import networks


validation_dataset = DatasetLoaderParser('./data/valid')
validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=32, shuffle=True,
        num_workers=12, pin_memory=True, sampler=None)

test_dataset = TestDataLoaderParser('./data/test')
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True,
        num_workers=12, pin_memory=True, sampler=None)

# pre and post processing variables
char_vocab = validation_dataset.char2idx
idx2char = validation_dataset.idx2chr
pad_idx = char_vocab[validation_dataset.pad_symbol]

# define device and network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = networks.DeepSpeech(vocab_size=len(char_vocab))

# load the pre-trained model

if device.type == 'cpu':
    model.load_state_dict(torch.load("deep_speech.pth", map_location='cpu'))
else:
    model.load_state_dict(torch.load("deep_speech.pth"))

model.to(device)

model.eval()
with torch.no_grad():

    # evaluate cer on validation set
    total_cer = 0
    num_validation_samples = len(validation_dataset)
    for i, data in enumerate(validation_loader, 1):
        inputs, labels = data[0].to(device), data[1].to(device)

        # greedy prediction
        output = model(inputs)
        _, predicted = torch.max(output, dim=2)
        predicted = predicted.permute(1, 0)  # transform to dimension (Batch, Timestamps)

        # post processing output
        out_transcript = validation_dataset.vectorized_idx2chr(predicted.cpu().numpy())
        labels = validation_dataset.vectorized_idx2chr(labels.cpu().numpy())

        total_cer += validation_dataset.get_batch_cer(out_transcript, labels)  # compare predicted and label transcripts

    validation_cer = total_cer / num_validation_samples
    print(f"Validation set CER is: {validation_cer}")

    # predict for test set
    with open("test_y.txt", "w") as f:
        num_files = 0
        for data in test_loader:
            file_paths, inputs = data[0], data[1].to(device)

            # greedy prediction
            output = model(inputs)
            _, predicted = torch.max(output, dim=2)
            predicted = predicted.permute(1, 0)  # transform to dimension (Batch, Timestamps)
            batch_size = predicted.size()[0]

            # post processing and writing to file
            out_transcript = validation_dataset.vectorized_idx2chr(predicted.cpu().numpy())

            for i in range(batch_size):
                file_name = file_paths[i]
                predicted_transcript = test_dataset.transcript_postprocessing(out_transcript[i])
                f.write(f"{file_name}, {predicted_transcript}\n")
                num_files += 1

    print(f"Predicted results for {num_files} files")
