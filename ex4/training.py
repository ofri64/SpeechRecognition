import torch
import torch.nn as nn
import torch.optim as optim

from gcommand_loader import GCommandLoaderSeqLabels
from networks import DeepSpeech

train_dataset = GCommandLoaderSeqLabels('./data/train')
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=None,
        num_workers=12, pin_memory=True, sampler=None)

validation_dataset = GCommandLoaderSeqLabels('./data/valid')
validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=32, shuffle=None,
        num_workers=12, pin_memory=True, sampler=None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

char_vocab = train_dataset.char2idx
pad_idx = char_vocab["<pad>"]
model = DeepSpeech(vocab_size=len(char_vocab)).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
ctc_loss = nn.CTCLoss()
epochs = 20

for epoch in range(epochs):

    epoch_loss = 0
    for data in train_loader:
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
        print(loss.item())
        epoch_loss += loss.item()


