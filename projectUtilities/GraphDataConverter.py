import numpy as np
import torch
from torch.utils.data import Dataset
import random


class GraphDataset(Dataset):
    def __init__(self, filepath):
        self.humanReadableHorseData = []

        source = open(filepath, 'r')
        previous_move = ""
        for number, line in enumerate(source):
            self.humanReadableHorseData.append(line.split(" "))
            pass
        pass

    def __len__(self):
        return len(self.humanReadableHorseData)

    def input_to_tensor(self, readable):
        value = [
            # FROM
            1.0 if readable[0] == 'A' else 0.0,
            1.0 if readable[0] == 'F' else 0.0,
            1.0 if readable[0] == 'D' else 0.0,
            1.0 if readable[0] == 'K' else 0.0,
            1.0 if readable[0] == 'P' else 0.0,
            1.0 if readable[0] == 'L' else 0.0,
            1.0 if readable[0] == 'V' else 0.0,
            1.0 if readable[0] == 'B' else 0.0,
            1.0 if readable[0] == 'X' else 0.0,
            1.0 if readable[0] == 'E' else 0.0,
            1.0 if readable[0] == 'R' else 0.0,
            1.0 if readable[0] == 'I' else 0.0,
            1.0 if readable[0] == 'S' else 0.0,
            1.0 if readable[0] == 'M' else 0.0,
            1.0 if readable[0] == 'G' else 0.0,
            1.0 if readable[0] == 'H' else 0.0,
            1.0 if readable[0] == 'C' else 0.0,

            # TO
            1.0 if readable[1] == 'A' else 0.0,
            1.0 if readable[1] == 'F' else 0.0,
            1.0 if readable[1] == 'D' else 0.0,
            1.0 if readable[1] == 'K' else 0.0,
            1.0 if readable[1] == 'P' else 0.0,
            1.0 if readable[1] == 'L' else 0.0,
            1.0 if readable[1] == 'V' else 0.0,
            1.0 if readable[1] == 'B' else 0.0,
            1.0 if readable[1] == 'X' else 0.0,
            1.0 if readable[1] == 'E' else 0.0,
            1.0 if readable[1] == 'R' else 0.0,
            1.0 if readable[1] == 'I' else 0.0,
            1.0 if readable[1] == 'S' else 0.0,
            1.0 if readable[1] == 'M' else 0.0,
            1.0 if readable[1] == 'G' else 0.0,
            1.0 if readable[1] == 'H' else 0.0,
            1.0 if readable[1] == 'C' else 0.0,
        ]

        return torch.tensor(value)
        pass

    def output_to_tensor(self, readable: str):
        value = [
            # FROM
            1.0 if readable.__contains__('A') else 0.0,
            1.0 if readable.__contains__('F') else 0.0,
            1.0 if readable.__contains__('D') else 0.0,
            1.0 if readable.__contains__('K') else 0.0,
            1.0 if readable.__contains__('P') else 0.0,
            1.0 if readable.__contains__('L') else 0.0,
            1.0 if readable.__contains__('V') else 0.0,
            1.0 if readable.__contains__('B') else 0.0,
            1.0 if readable.__contains__('X') else 0.0,
            1.0 if readable.__contains__('E') else 0.0,
            1.0 if readable.__contains__('R') else 0.0,
            1.0 if readable.__contains__('I') else 0.0,
            1.0 if readable.__contains__('S') else 0.0,
            1.0 if readable.__contains__('M') else 0.0,
            1.0 if readable.__contains__('G') else 0.0,
            1.0 if readable.__contains__('H') else 0.0,
            1.0 if readable.__contains__('C') else 0.0,
        ]

        return torch.tensor(value)
        pass

    def tensor_to_output_best(self, tensor):
        from_value = tensor[0:17].detach().numpy()
        from_value_best = "AFDKPLVBXERISMGHC"[np.where(from_value == np.max(from_value))[0][0]]
        return from_value_best
        pass

    def tensor_to_output_top_3_best(self, tensor):
        from_value = tensor[0:17].detach().numpy()
        random_to_top_3 = random.choice((np.sort(from_value)[::-1])[:3])
        from_value_best = "AFDKPLVBXERISMGHC"[np.where(from_value == random_to_top_3)[0][0]]
        return from_value_best
        pass

    def __getitem__(self, idx):
        pair = self.humanReadableHorseData[idx]
        return self.input_to_tensor(pair[0]), self.output_to_tensor(pair[1])

    pass
