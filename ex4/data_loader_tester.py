from gcommand_loader import GCommandLoader, GCommandLoaderSeqLabels
import torch

dataset = GCommandLoaderSeqLabels('./data/valid')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

for k, (input, label) in enumerate(test_loader):
    if k < 10:
        print(input.size(), label.size())
    # input is of size N, C, F, T -
    # where N is the batch size
    # C is the number of "channels" for the signal (just 1 for the input layer)
    # F is num of frequencies in histogram
    # T is number of time frames (padded or pruned to 101)

    # label is of size N, S
    # where N is the batch size
    # S is the longest transcriptions label length
