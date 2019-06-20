import torch
import torch.nn as nn
import torch.optim as optim

from gcommand_loader import DatasetLoaderParser
import networks
from cer import cer


def define_model_computing_conf(model_, single_batch_size=32):
    device_ = None
    batch_size_ = single_batch_size
    if not torch.cuda.is_available():  # only CPU
        device_ = torch.device("cpu")
    elif torch.cuda.device_count() == 1:  # single GPU
        device_ = torch.device("cuda:0")
    else:  # multiple GPUs
        model_ = nn.DataParallel(model_)
        batch_size_ = single_batch_size * torch.cuda.device_count()

    return model_.to(device_), device_, batch_size_


# load train and validation datasets
train_dataset = DatasetLoaderParser('./data/train')
validation_dataset = DatasetLoaderParser('./data/valid')

# pre and post processing variables
char_vocab = train_dataset.char2idx
idx2char = train_dataset.idx2chr
pad_idx = char_vocab[train_dataset.pad_symbol]

# define device. network and training properties
model = networks.DeeperDeepSpeech(vocab_size=len(char_vocab))
model, device, batch_size = define_model_computing_conf(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss = nn.CTCLoss()
num_epochs = 80
lowest_validation_cer = 10000

# prepare training and validation loaders
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, sampler=None)

validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, sampler=None)

for epoch in range(num_epochs):

    # switch to train mode
    model.train(mode=True)

    epoch_loss = 0
    running_loss = 0
    for i, data in enumerate(train_loader, 1):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)

        # compute inputs_length and original target length (without padding)
        T, N, F_ = outputs.size()
        inputs_length = torch.full(size=(N,), fill_value=T, dtype=torch.int32).to(device)
        targets_length = torch.sum(labels != pad_idx, dim=1).to(device)

        # ctc loss + backward + optimize
        loss = ctc_loss(outputs, labels, inputs_length, targets_length)
        loss.backward()
        optimizer.step()

        # loss statistics
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch + 1} Loss is: {epoch_loss}")


# evaluate using a greedy tagger over training and validation sets
# model.load_state_dict(torch.load("deep_speech.pth"))
# model.to(device)

    model.eval()
    with torch.no_grad():

        total_cer = 0
        num_train_samples = len(train_dataset)
        for i, data in enumerate(train_loader, 1):
            inputs, labels = data[0].to(device), data[1].to(device)

            # greedy prediction
            output = model(inputs)
            _, predicted = torch.max(output, dim=2)
            predicted = predicted.permute(1, 0)  # transform to dimension (Batch, Timestamps)

            # post processing output
            out_transcript = train_dataset.vectorized_idx2chr(predicted.cpu().numpy())
            labels = train_dataset.vectorized_idx2chr(labels.cpu().numpy())

            total_cer += train_dataset.get_batch_cer(out_transcript, labels)  # compare predicted and label transcripts

            # print the output transcript and the true label every once in a while
            # if i % 10000 == 0:
            #     transcript = out_transcript[0]
            #     label = labels[0]
            #
            #     # post processing transcript and labels
            #     transcript = train_dataset.transcript_postprocessing(transcript)
            #     label_transcript = train_dataset.label_postprocessing(label)
            #     print(f"Model predicted transcript: \"{transcript}\", while gold label is: \"{label_transcript}\"")

        epoch_cer = total_cer / num_train_samples
        print(f"Epoch {epoch + 1}: Training set CER is: {epoch_cer}")

        # predict on validation set now

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

            # print the output transcript and the true label every once in a while
            if i % 20 == 0:
                transcript = out_transcript[0]
                label = labels[0]

                # post processing transcript and labels
                transcript = validation_dataset.transcript_postprocessing(transcript)
                label_transcript = validation_dataset.label_postprocessing(label)
                current_cer = cer(transcript, label_transcript)
                print(f"Model predicted transcript: \"{transcript}\", while gold label is: \"{label_transcript}\" - CER is: {current_cer}")

        epoch_cer = total_cer / num_validation_samples
        print(f"Epoch {epoch + 1}: Validation set CER is: {epoch_cer}")

        if epoch_cer < lowest_validation_cer:
            lowest_validation_cer = epoch_cer
            print(f"Epoch {epoch + 1} saving model parameters")
            torch.save(model.state_dict(), "deep_speech.pth")
