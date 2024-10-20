# import minitorch

# # Use this function to make a random parameter in
# # your module.
# def RParam(*shape):
#     # r = 2 * (minitorch.rand(shape) - 0.5)
#     r = minitorch.zeros(shape) + 1.0
#     return minitorch.Parameter(r)

# # TODO: Implement for Task 2.5.

# class Network(minitorch.Module):
#     def __init__(self, hidden_layers):
#         super().__init__()
#         # self.layer1 = Linear(2, hidden_layers)
#         # self.layer2 = Linear(hidden_layers, hidden_layers)
#         # self.layer3 = Linear(hidden_layers, 1)

#         self.layer1 = Linear(2, hidden_layers)

#     def forward(self, x):
#         # middle = self.layer1.forward(x).relu()
#         # end = self.layer2.forward(middle).relu()
#         # return self.layer3.forward(end).sigmoid()

#         return self.layer1.forward(x).sigmoid()


# class Linear(minitorch.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.weights = RParam(in_size, out_size)
#         self.bias = RParam(out_size)
#         self.out_size = out_size


#     def forward(self, inputs):
#         """weight matrix mult input then add bias, output a tensor"""
#         # type Parameter .value is the tensor
#         # inputs = list of tensors, each representing a data point initially
#         # return self.weights.value * inputs + self.bias.value

#         # initially call fwd on dataset which has shape (num_pts, 2) = (num_pts, in_size) for first layer
#         num_pts, in_size = inputs.shape
#         # first layer weight has shape (in_size, out_size) = (2, hidden_layers)
#         # want to broadcast so that (1, in_size, out_size) and (num_pts, in_size, 1)
#         matix_mul = (self.weights.value.view(1, in_size, self.out_size) * inputs.view(num_pts, in_size, 1)).sum(1).view(num_pts, self.out_size)
#         # result is (num_pts, out_size), i.e. one output for each datapoint

#         return matix_mul + self.bias.value.view(1, self.out_size)



# def default_log_fn(epoch, total_loss, correct, losses):
#     print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


# class TensorTrain:
#     def __init__(self, hidden_layers):
#         self.hidden_layers = hidden_layers
#         self.model = Network(hidden_layers)

#     def run_one(self, x):
#         return self.model.forward(minitorch.tensor([x]))

#     def run_many(self, X):
#         return self.model.forward(minitorch.tensor(X))

#     def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
#         self.learning_rate = learning_rate
#         self.max_epochs = max_epochs
#         self.model = Network(self.hidden_layers)
#         optim = minitorch.SGD(self.model.parameters(), learning_rate)

#         X = minitorch.tensor(data.X)
#         y = minitorch.tensor(data.y)

#         losses = []
#         for epoch in range(1, self.max_epochs + 1):
#             total_loss = 0.0
#             correct = 0
#             optim.zero_grad()

#             # Forward
#             out = self.model.forward(X).view(data.N)
#             prob = (out * y) + (out - 1.0) * (y - 1.0)

#             loss = -prob.log()
#             (loss / data.N).sum().view(1).backward()
#             total_loss = loss.sum().view(1)[0]
#             losses.append(total_loss)

#             # Update
#             optim.step()

#             # Logging
#             if epoch % 10 == 0 or epoch == max_epochs:
#                 y2 = minitorch.tensor(data.y)
#                 correct = int(((out.detach() > 0.5) == y2).sum()[0])
#                 log_fn(epoch, total_loss, correct, losses)


# import torch
# class TorchModel(torch.nn.Module):

#     def __init__(self, hidden_layers):
#         super(TorchModel, self).__init__()
#         self.linear1 = torch.nn.Linear(2, hidden_layers)
#         self.linear1.weight.data = torch.ones(2, hidden_layers)

#         # self.linear2 = torch.nn.Linear(hidden_layers, hidden_layers)
#         # self.linear3 = torch.nn.Linear(hidden_layers, 1)
#         self.activation = torch.nn.ReLU()
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.linear1(x)
#         # x = self.activation(x)
#         # x = self.linear2(x)
#         # x = self.activation(x)
#         # x = self.linear3(x)
#         x = self.sigmoid(x)
#         return x

# # def testMatMul():
# #     # (1, in_size, out_size)
# #     # (num_pts, in_size, 1)

# if __name__ == "__main__":
#     PTS = 50
#     HIDDEN = 2
#     RATE = 0.5
#     data = minitorch.datasets["Simple"](PTS)

#     myModel = Network(2)
#     pyModel = TorchModel(2)
#     myOut = myModel.forward(minitorch.tensor(data.X))
#     pyOut = pyModel.forward(torch.tensor(data.X))
#     print(myOut)
#     print(pyOut)