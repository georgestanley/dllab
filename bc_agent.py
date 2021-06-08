import torch
from agent.networks import CNN

# from networks import  CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device = ", device)


class BCAgent:

    def __init__(self, history_length=0, lr = 0.001):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length, n_classes=5).to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        self.net.train()
        X_batch = torch.from_numpy(X_batch)
        X_batch = X_batch.to(device, dtype = torch.float)
        print("X_batch shape", X_batch.shape)

        y_batch = torch.from_numpy(y_batch)
        y_batch = y_batch.to(device, dtype=torch.long)
        print("Y_batch shape", y_batch.shape)

        print("transforms done")

        # TODO: forward + backward + optimize

        outputs = self.net(X_batch)
        outputs = torch.squeeze(outputs)
        loss = self.loss_fn(outputs, y_batch)
        #print("shape of outputs", outputs.shape, "shape of y_batch", y_batch.shape)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total = len(y_batch)
        correct = (predicted==y_batch).sum().item()
        acc = (100*correct)/total
        #acc = torch.round(outputs).eq(y_batch).float().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), acc

    def predict_val(self, X, y):
        # TODO: forward pass
        self.net.eval()
        X = torch.from_numpy(X)
        X = X.to(device, dtype = torch.float)
        y = torch.from_numpy(y)
        y = y.to(device, dtype=torch.long)
        outputs = self.net(X)
        outputs = torch.squeeze(outputs)
        print("output shape",outputs.shape)
        loss = self.loss_fn(outputs, y)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total = len(y)
        correct = (predicted == y).sum().item()
        acc = (100 * correct) / total

        # return outputs
        return loss.item(), acc

    def predict(self, X):
        self.net.eval()
        X = torch.from_numpy(X)
        X = X.to(device, dtype = torch.float)
        outputs = self.net(X)
        outputs = torch.squeeze(outputs)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
