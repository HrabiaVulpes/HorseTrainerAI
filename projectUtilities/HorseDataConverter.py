import numpy as np
import torch
from torch.utils.data import Dataset
import random


class HorseDataDataset(Dataset):
    def __init__(self, filepath):
        self.humanReadableHorseData = []

        source = open(filepath, 'r')
        previous_move = ""
        for number, line in enumerate(source):
            line = line.strip()
            if line == "":
                previous_move = ""
                continue
            if previous_move != "":
                self.humanReadableHorseData.append([previous_move, [line]])
                pass

            assert "AFDKPLVBXERISMGHC".__contains__(line[0]), f"Wrong letter 0 in line {number}"
            assert "AFDKPLVBXERISMGHC".__contains__(line[1]), f"Wrong letter 1 in line {number}"
            assert "XWVITYGCOH".__contains__(line[2]), f"Wrong letter 2 in line {number}"
            assert "LCFBGSTDPAZH".__contains__(line[3]), f"Wrong letter 3 in line {number}"
            previous_move = line
            pass
        self.dataexpansion()
        pass

    def dataexpansion(self):
        for i in range(len(self.humanReadableHorseData)):
            recipient = self.humanReadableHorseData[i]
            for j in range(len(self.humanReadableHorseData)):
                source = self.humanReadableHorseData[j]
                if i != j:
                    if self.humanReadableHorseData[i][0][0] == self.humanReadableHorseData[j][0][0]:
                        if self.humanReadableHorseData[i][0][1] == self.humanReadableHorseData[j][0][1]:
                            self.humanReadableHorseData[i][1].extend(self.humanReadableHorseData[j][1])
                            pass
                pass
            pass
        for i in range(len(self.humanReadableHorseData)):
            self.humanReadableHorseData[i][1] = list(set(self.humanReadableHorseData[i][1]))
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

            # # TO
            # 1.0 if readable[1] == 'A' else 0.0,
            # 1.0 if readable[1] == 'F' else 0.0,
            # 1.0 if readable[1] == 'D' else 0.0,
            # 1.0 if readable[1] == 'K' else 0.0,
            # 1.0 if readable[1] == 'P' else 0.0,
            # 1.0 if readable[1] == 'L' else 0.0,
            # 1.0 if readable[1] == 'V' else 0.0,
            # 1.0 if readable[1] == 'B' else 0.0,
            # 1.0 if readable[1] == 'X' else 0.0,
            # 1.0 if readable[1] == 'E' else 0.0,
            # 1.0 if readable[1] == 'R' else 0.0,
            # 1.0 if readable[1] == 'I' else 0.0,
            # 1.0 if readable[1] == 'S' else 0.0,
            # 1.0 if readable[1] == 'M' else 0.0,
            # 1.0 if readable[1] == 'G' else 0.0,
            # 1.0 if readable[1] == 'H' else 0.0,
            # 1.0 if readable[1] == 'C' else 0.0,

            # # SPEED
            # 1.0 if readable[2] == 'X' else 0.0,
            # 1.0 if readable[2] == 'W' else 0.0,
            # 1.0 if readable[2] == 'V' else 0.0,
            # 1.0 if readable[2] == 'I' else 0.0,
            # 1.0 if readable[2] == 'T' else 0.0,
            # 1.0 if readable[2] == 'Y' else 0.0,
            # 1.0 if readable[2] == 'G' else 0.0,
            # 1.0 if readable[2] == 'C' else 0.0,
            # 1.0 if readable[2] == 'O' else 0.0,
            # 1.0 if readable[2] == 'H' else 0.0,
            #
            # # SHAPE
            # 1.0 if readable[3] == 'L' else 0.0,
            # 1.0 if readable[3] == 'C' else 0.0,
            # 1.0 if readable[3] == 'F' else 0.0,
            # 1.0 if readable[3] == 'B' else 0.0,
            # 1.0 if readable[3] == 'G' else 0.0,
            # 1.0 if readable[3] == 'S' else 0.0,
            # 1.0 if readable[3] == 'T' else 0.0,
            # 1.0 if readable[3] == 'D' else 0.0,
            # 1.0 if readable[3] == 'P' else 0.0,
            # 1.0 if readable[3] == 'A' else 0.0,
            # 1.0 if readable[3] == 'Z' else 0.0,
            # 1.0 if readable[3] == 'H' else 0.0
        ]

        return torch.tensor(value)
        pass

    def list_output_to_tensor(self, readable_list):
        final_tensor = torch.zeros(self.output_to_tensor(readable_list[0]).shape[0])
        for element in readable_list:
            tensor_element = self.output_to_tensor(element)
            if not torch.isfinite(tensor_element).all():
                print("NaNs in model input parts")
            final_tensor = final_tensor.add(tensor_element)
            pass
        if not torch.isfinite(final_tensor).all():
            print("NaNs in model input")
        return final_tensor

    def output_to_tensor(self, readable):
        value = [
            # FROM
            # 1.0 if readable[0] == 'A' else 0.0,
            # 1.0 if readable[0] == 'F' else 0.0,
            # 1.0 if readable[0] == 'D' else 0.0,
            # 1.0 if readable[0] == 'K' else 0.0,
            # 1.0 if readable[0] == 'P' else 0.0,
            # 1.0 if readable[0] == 'L' else 0.0,
            # 1.0 if readable[0] == 'V' else 0.0,
            # 1.0 if readable[0] == 'B' else 0.0,
            # 1.0 if readable[0] == 'X' else 0.0,
            # 1.0 if readable[0] == 'E' else 0.0,
            # 1.0 if readable[0] == 'R' else 0.0,
            # 1.0 if readable[0] == 'I' else 0.0,
            # 1.0 if readable[0] == 'S' else 0.0,
            # 1.0 if readable[0] == 'M' else 0.0,
            # 1.0 if readable[0] == 'G' else 0.0,
            # 1.0 if readable[0] == 'H' else 0.0,
            # 1.0 if readable[0] == 'C' else 0.0,

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

            # # SPEED
            # 1.0 if readable[2] == 'X' else 0.0,
            # 1.0 if readable[2] == 'W' else 0.0,
            # 1.0 if readable[2] == 'V' else 0.0,
            # 1.0 if readable[2] == 'I' else 0.0,
            # 1.0 if readable[2] == 'T' else 0.0,
            # 1.0 if readable[2] == 'Y' else 0.0,
            # 1.0 if readable[2] == 'G' else 0.0,
            # 1.0 if readable[2] == 'C' else 0.0,
            # 1.0 if readable[2] == 'O' else 0.0,
            # 1.0 if readable[2] == 'H' else 0.0,
            #
            # # SHAPE
            # 1.0 if readable[3] == 'L' else 0.0,
            # 1.0 if readable[3] == 'C' else 0.0,
            # 1.0 if readable[3] == 'F' else 0.0,
            # 1.0 if readable[3] == 'B' else 0.0,
            # 1.0 if readable[3] == 'G' else 0.0,
            # 1.0 if readable[3] == 'S' else 0.0,
            # 1.0 if readable[3] == 'T' else 0.0,
            # 1.0 if readable[3] == 'D' else 0.0,
            # 1.0 if readable[3] == 'P' else 0.0,
            # 1.0 if readable[3] == 'A' else 0.0,
            # 1.0 if readable[3] == 'Z' else 0.0,
            # 1.0 if readable[3] == 'H' else 0.0
        ]

        return torch.tensor(value)
        pass

    def tensor_to_output_best(self, tensor):
        from_value = tensor[0:17].detach().numpy()
        # to_value = tensor[17:2 * 17].detach().numpy()
        # speed_value = tensor[2 * 17: 2 * 17 + 10].detach().numpy()
        # shape_value = tensor[2 * 17 + 10: 2 * 17 + 10 + 12].detach().numpy()

        from_value_best = "AFDKPLVBXERISMGHC"[np.where(from_value == np.max(from_value))[0][0]]
        # to_value_best = "AFDKPLVBXERISMGHC"[np.where(to_value == np.max(to_value))[0][0]]
        # speed_value_best = "XWVITYGCOH"[np.where(speed_value == np.max(speed_value))[0][0]]
        # shape_value_best = "LCFBGSTDPAZH"[np.where(shape_value == np.max(shape_value))[0][0]]

        return from_value_best# + to_value_best  # , speed_value_best, shape_value_best
        pass

    def tensor_to_output_top_3_best(self, tensor):
        from_value = tensor[0:17].detach().numpy()
        # to_value = tensor[17:2 * 17].detach().numpy()
        # speed_value = tensor[2 * 17: 2 * 17 + 10].detach().numpy()
        # shape_value = tensor[2 * 17 + 10: 2 * 17 + 10 + 12].detach().numpy()

        # max_from = np.max(from_value)
        # random_to_top_3 = random.choice((np.sort(to_value)[::-1])[:3])
        # random_speed_top_3 = random.choice(np.sort(speed_value)[:3])
        # random_shape_top_3 = random.choice(np.sort(shape_value)[:3])

        random_to_top_3 = random.choice((np.sort(from_value)[::-1])[:3])

        to_value_best = "AFDKPLVBXERISMGHC"[np.where(from_value == random_to_top_3)[0][0]]
        # from_value_best = "AFDKPLVBXERISMGHC"[np.where(from_value == max_from)[0][0]]
        # to_value_best = "AFDKPLVBXERISMGHC"[np.where(to_value == random_to_top_3)[0][0]]
        # speed_value_best = "XWVITYGCOH"[np.where(speed_value == random_speed_top_3)[0][0]]
        # shape_value_best = "LCFBGSTDPAZH"[np.where(shape_value == random_shape_top_3)[0][0]]

        return to_value_best  # , speed_value_best, shape_value_best
        pass

    def __getitem__(self, idx):
        pair = self.humanReadableHorseData[idx]
        return self.input_to_tensor(pair[0]), self.list_output_to_tensor(pair[1])

    pass
