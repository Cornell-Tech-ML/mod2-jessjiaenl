"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import torch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    r = minitorch.zeros(shape) + 0.8
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        # self.layer2 = Linear(hidden_layers, hidden_layers)
        # self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # middle = self.layer1.forward(x).relu()
        # end = self.layer2.forward(middle).relu()
        # output = self.layer3.forward(end).sigmoid()
        # return output

        return self.layer1.forward(x).relu()

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size


    def forward(self, inputs):
        num_pts, in_size = inputs.shape
        # first layer weight has shape (in_size, out_size) = (2, hidden_layers)
        # want to broadcast so that (1, in_size, out_size) and (num_pts, in_size, 1)
        w = self.weights.value # (1, in_size, out_size)
        xs = inputs.view(num_pts, in_size, 1)
        matix_mul = (w*xs).sum(1).view(num_pts, self.out_size)
        # result is (num_pts, out_size), i.e. one output for each datapoint
        return matix_mul + self.bias.value # bias shape is (1, out_size)



class PyTorchLinear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(in_size, out_size) - 0.2)
        self.bias = torch.nn.Parameter(torch.ones(out_size)-0.2)

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.bias

class PyTorchNetwork(torch.nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        # self.layer1 = PyTorchLinear(2, hidden_layers)
        # self.layer2 = PyTorchLinear(hidden_layers, hidden_layers)
        # self.layer3 = PyTorchLinear(hidden_layers, 1)
        self.layer1 = PyTorchLinear(2, hidden_layers)

    def forward(self, x):
        # middle = torch.relu(self.layer1(x))
        # end = torch.relu(self.layer2(middle))
        # output = torch.sigmoid(self.layer3(end))
        # return output
        return torch.relu(self.layer1(x))



def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 10
    HIDDEN = 2
    RATE = 0.1
    data = minitorch.datasets["Simple"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    # Create and run MiniTorch model
    minitorch_model = Network(HIDDEN)
    minitorch_input = minitorch.tensor(data.X)
    minitorch_output = minitorch_model.forward(minitorch_input)
    # print(minitorch_output)

    minitorch_loss = minitorch_output.sum()
    minitorch_loss.backward()

    print(f"Minitorch Loss: {minitorch_loss}")

    # print(minitorch_loss.grad)
    # print(minitorch_model.grad)

    print(minitorch_model.named_parameters)
    print(minitorch_model.layer1)
    print("LAYER 1 GRAD")
    print(minitorch_model.layer1.weights.value.grad)
    print(minitorch_model.layer1.bias.value.grad)


    # Create and run PyTorch model
    pytorch_model = PyTorchNetwork(HIDDEN)
    pytorch_input = torch.tensor(data.X, dtype=torch.float32)
    pytorch_output = pytorch_model(pytorch_input)
    # print(pytorch_output)

    pytorch_loss = pytorch_output.sum()
    pytorch_loss.backward()

    print(f"Pytorch Loss: {pytorch_loss}")

    print(pytorch_model.named_parameters)
    print(pytorch_model.layer1)
    print("LAYER 1 GRAD")
    print(pytorch_model.layer1.weights.grad)
    print(pytorch_model.layer1.bias.grad)


    # for name, param in pytorch_model.named_parameters():
    #     print(f"{name} grad: {param.grad}")