import torch
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, orderedInputData, orderedOutputData, inputTransform=None, outputTransform=None):
        self.inputData = orderedInputData
        self.outputData = orderedOutputData
        self.inputTransform = inputTransform
        self.outputTransform = outputTransform

    def __len__(self):
        return len(self.inputData)

    def __getitem__(self, idx):
        input = self.inputData[idx]
        output = self.outputData[idx]

        if self.inputTransform:
            input = self.inputTransform(input)

        if self.outputTransform:
            output = self.outputTransform(output)

        return input, output
