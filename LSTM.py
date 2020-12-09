import sys
import numpy as np
import torch
import torch.nn as nn


class LSTM_network(nn.Module):
    """ Define LSTM
    """
    def __init__(self, params, device):
        super(LSTM_network, self).__init__()
        self.params = params
        self.device = device

        self.lstm = nn.LSTM(input_size=2, hidden_size=params["num_features"])
        self.linear1 = nn.Linear(params["num_features"], params["num_features"])
        self.linear2 = nn.Linear(params["num_features"], 2)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights
        """
        self.lstm.weight_ih_l0.data.normal_(0, 0.001)
        self.lstm.weight_hh_l0.data.normal_(0, 0.001)
        self.linear1.weight.data.normal_(0, 0.001)
        self.linear2.weight.data.normal_(0, 0.001)

    def forward(self, x_in):
        # lstm -> Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        # lstm INPUT -> 3D tensors: first axis is the sequence, 
        # second is instances in the minibatch, third is indexes of the input
        x, self.hidden_cell = self.lstm(x_in, self.hidden_cell)
        x = self.linear1(x)
        x = self.linear2(x)[-1]
        return x

    def forward_predict(self, x):
        self.hidden_cell = (torch.zeros(1, x.shape[1], self.params["num_features"]).to(self.device),
                                  torch.zeros(1, x.shape[1], self.params["num_features"]).to(self.device))
        x_out = torch.zeros([x.shape[0]+self.params["output_window"], x.shape[1], x.shape[2]]).to(self.device)
        x_out[:x.shape[0], :, :] = x
        for idx in range(x.shape[0], self.params["output_window"]+x.shape[0]):
            x_out[idx, :, :] = self.forward(x_out[idx-x.shape[0]:idx, :, :])
        return x_out[x.shape[0]:, :, :]


class LSTM():
    def __init__(self, params, device ="cuda" if torch.cuda.is_available() else "cpu"):
        self.params = params
        self.device = device
        # Reproducability
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Model
        self.model = LSTM_network(self.params, self.device)
        self.model.to(self.device)
        # Optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.params["learning_rate"])

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            if self.device == "cuda":
                return self.model.forward_predict(torch.tensor(inputs, dtype=torch.float32).permute(2,0,1).to(self.device)).cpu().permute(1,2,0).numpy()
            else:
                return self.model.forward_predict(torch.tensor(inputs, dtype=torch.float32).permute(2,0,1)).permute(1,2,0).numpy()


    def train(self, train_loader, val_loader, test_loader, verbose=False):
        """
        Train, vlaidate and test the LSTM. The best results are stored/
        """
        best_epoch = 0
        best_train_loss = sys.maxsize
        best_val_loss = sys.maxsize
        best_test_loss = sys.maxsize
        for epoch in range(1 + self.params["epochs"]):
            # TRAIN
            self.model.train()
            #self.model.load_state_dict(torch.load("data/lstm_pretrained.pt", map_location=torch.device(DEVICE)))
            train_loss = []
            for inputs, targets in train_loader:
                inputs = inputs.permute(2,0,1).to(self.device)
                targets = targets.permute(2,0,1).to(self.device)
                self.optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, inputs.shape[1], self.params["num_features"]).to(self.device),
                                          torch.zeros(1, inputs.shape[1], self.params["num_features"]).to(self.device))
                outputs = self.model(inputs)
                loss = nn.functional.mse_loss(outputs, targets[0, :, :]) / self.params["output_window"]
                for idx in range(1, self.params["output_window"]):
                    inputs = torch.cat((inputs, outputs.unsqueeze(0)),0)[1:,:,:].to(self.device)
                    outputs = self.model(inputs)
                    loss += nn.functional.mse_loss(outputs, targets[idx, :, :]) / self.params["output_window"]
                if epoch > 0:
                    loss.backward()
                    self.optimizer.step()
                [train_loss.append(loss.item()) for i in range(inputs.shape[0])]
            train_loss = np.mean(train_loss)
            # VAL & TEST
            self.model.eval()
            with torch.no_grad():
                val_loss = []
                for inputs, targets in val_loader:
                    inputs = inputs.permute(2,0,1).to(self.device)
                    targets = targets.permute(2,0,1).to(self.device)
                    self.model.hidden_cell = (torch.zeros(1, inputs.shape[1], self.params["num_features"]).to(self.device),
                                              torch.zeros(1, inputs.shape[1], self.params["num_features"]).to(self.device))
                    outputs = self.model(inputs)
                    loss = nn.functional.mse_loss(outputs, targets[0, :, :]) / self.params["output_window"]
                    for idx in range(1, self.params["output_window"]):
                        inputs = torch.cat((inputs, outputs.unsqueeze(0)),0)[1:,:,:].to(self.device)
                        outputs = self.model(inputs)
                        loss += nn.functional.mse_loss(outputs, targets[idx, :, :]) / self.params["output_window"]
                    [val_loss.append(loss.item()) for i in range(inputs.shape[0])]
                val_loss = np.mean(val_loss)
                test_loss = []
                test_target = []
                test_output = []
                for inputs, targets in test_loader:
                    inputs = inputs.permute(2,0,1).to(self.device)
                    targets = targets.permute(2,0,1).to(self.device)
                    test_output_intermediate = []
                    self.model.hidden_cell = (torch.zeros(1, inputs.shape[1], self.params["num_features"]).to(self.device),
                                              torch.zeros(1, inputs.shape[1], self.params["num_features"]).to(self.device))
                    outputs = self.model(inputs)
                    test_target.append(targets)
                    test_output_intermediate.append(outputs)
                    loss = nn.functional.mse_loss(outputs, targets[0, :, :]) / self.params["output_window"]
                    for idx in range(1, self.params["output_window"]):
                        inputs = torch.cat((inputs, outputs.unsqueeze(0)),0)[1:,:,:].to(self.device)
                        outputs = self.model(inputs)
                        test_output_intermediate.append(outputs)
                        loss += nn.functional.mse_loss(outputs, targets[idx, :, :]) / self.params["output_window"]
                    test_output.append(torch.stack(test_output_intermediate))
                    [test_loss.append(loss.item()) for i in range(inputs.shape[0])]

                test_loss = np.mean(test_loss)
            # Save best results
            if val_loss < best_val_loss:
                best_epoch = epoch
                best_train_loss = train_loss
                best_val_loss = val_loss
                best_test_loss = test_loss
            # Verbose
            if verbose:
                print(f"Epoch:{epoch}, Train:{train_loss}, Val:{val_loss}, Test:{test_loss}")
        return best_train_loss, best_val_loss, best_test_loss


if __name__ == "__main__":
    pass
