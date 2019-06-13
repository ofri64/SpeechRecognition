import torch
import torch.nn as nn
import torch.optim as optim

from gcommand_loader import DatasetLoaderParser
from cer import cer
from networks import DeepSpeech

# load train and validation datasets

train_dataset = DatasetLoaderParser('./data/train')
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=None,
        num_workers=12, pin_memory=True, sampler=None)

train_loader_single_sample = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=None,
        num_workers=12, pin_memory=True, sampler=None)

validation_dataset = DatasetLoaderParser('./data/valid')
validation_loader_single_sample = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=None,
        num_workers=12, pin_memory=True, sampler=None)

# pre and post processing variables
char_vocab = train_dataset.char2idx
idx2char = train_dataset.idx2chr
pad_idx = char_vocab[train_dataset.pad_symbol]

# define device. network and training properties
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DeepSpeech(vocab_size=len(char_vocab)).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
ctc_loss = nn.CTCLoss()
num_epochs = 3

for epoch in range(num_epochs):

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
        running_loss += loss.item()
        if i % 200 == 0:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / 200))
            running_loss = 0.0

    # evaluate using a greedy tagger over training and validation sets
    with torch.no_grad():
        total_cer = 0
        num_train_samples = len(train_loader_single_sample)
        for data in train_loader_single_sample:
            input_, label = data[0].to(device), data[1].to(device)

            # greedy prediction
            output = model(input_)
            _, predicted = torch.max(output, dim=2)

            # post processing output

            label = label.flatten()
            label = label[label != pad_idx]  # remove padding
            label_transcript = "".join([idx2char[idx] for idx in label.numpy()])  # convert back to string

            blank_transcript = "".join([idx2char[idx] for idx in predicted.flatten().numpy()])
            out_transcript = DatasetLoaderParser.transcript_postprocessing(blank_transcript)  # remove blanks

            total_cer += cer(out_transcript, label_transcript)  # compare predicted and label transcripts

        epoch_cer = total_cer / num_train_samples
        print(f"Epoch {epoch + 1}: Training set CER is: {epoch_cer}")