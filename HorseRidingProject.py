from projectUtilities.HorseDataConverter import HorseDataDataset
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
horse_rides_dataset = HorseDataDataset("resources/trainingData.txt")
horse_rides_dataloader = DataLoader(horse_rides_dataset, batch_size=32)

input_length = 0
output_length = 0

print(f"result: {horse_rides_dataset.humanReadableHorseData}")
print(f"result: {horse_rides_dataset.__getitem__(0)}")
for X, y in horse_rides_dataloader:
    print(f"Length on input: {X.shape}")
    input_length = X.shape[1]
    print(f"Length of output: {y.shape}")
    output_length = y.shape[1]
    break


class HorseRider(nn.Module):
    def __init__(self):
        super(HorseRider, self).__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(input_length, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_length),
            nn.ReLU()
        )
        pass

    def forward(self, x):
        return self.neural_network(x)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # initialize the weight tensor, here we use a normal distribution
                m.weight.data.normal_(0, 1)
                pass
            pass
        pass

    pass


neural_network_model = HorseRider()
print(neural_network_model)

loss_function = nn.MSELoss()
optimizer = optim.Adam(neural_network_model.parameters(), lr=0.01)


def train(dataloader, model, loss_fn, optimizer):
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        prediction = model(x)
        predicted = horse_rides_dataset.tensor_to_output_best(prediction[0])
        if not torch.isfinite(prediction).all():
            print("NaNs in model output")

        loss = loss_fn(prediction, y)
        expected = horse_rides_dataset.tensor_to_output_best(y[0])

        if not torch.isfinite(loss).all():
            print("NaNs in loss")

        loss.backward()
        if not all(torch.isfinite(param.grad).all() for param in model.parameters() if param.grad is not None):
            print("NaNs in gradients")
        optimizer.step()

        # print(f"Predicted = {predicted}, expected = {expected}, error = {loss.item()}")
        pass
    pass


training_runs = 10000
neural_network_model.train()
for i in range(training_runs):
    train(horse_rides_dataloader, neural_network_model, loss_function, optimizer)
    pass
neural_network_model.eval()

horse_from = "AD"
for i in range(5):
    effect = neural_network_model(horse_rides_dataset.input_to_tensor(horse_from))
    horse_to = horse_from[1] + horse_rides_dataset.tensor_to_output_top_3_best(effect)
    print(horse_from + " => " + horse_to)
    horse_from = horse_to
    pass
