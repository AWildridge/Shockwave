import os
import yaml
import argparse
from itertools import cycle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from src import data
from src import architectures
from src import plotting

# Parse model folder and wandb name
parser = argparse.ArgumentParser()
parser.add_argument('model_folder', help="The folder where the model's config is stored, and where everything will be saved.", type=str)
#parser.add_argument('wandb_name', help="Name for the run in wandb.", type=str)
args = parser.parse_args()
SAVE_FOLDER = args.model_folder # for convenience

# Detect GPU, train on CPU if not available
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
print("Will train on: " + DEVICE)
DEVICE = torch.device(DEVICE)

# Read in YAML config file
config_filename = SAVE_FOLDER + "/config.yaml"
with open(config_filename, 'r') as config_file:
        config = yaml.safe_load(config_file)
batch_size = config['dataset'].get('batch_size', 128) # Default batch size: 128
epochs = config['optimizer'].get('epochs', 20) # Default number of epochs: 20

# Load and preprocess dataset
datasets_folder = config['dataset'].get('location')
data_array = np.load(datasets_folder+"/background.npz")["data"]

# Trying to predict the second channel given the first one
# inputs_array = data_array[:, 0, :]
# targets_array = data_array[:, 1, :]

#train_dataset, valid_dataset, test_dataset = data.PrepareDatasets(data_array, seed=5138008)
#train_dataset, valid_dataset, test_dataset = data.PrepareTwoChannelSimpleDatasets(data_array, seed=5138008)
train_dataset, valid_dataset, test_dataset = data.PrepareFFTDatasets(data_array, seed=5138008)

# Create log file and write first line
info = open(SAVE_FOLDER+"/model_details.txt", 'w')
info.write("Size of training dataset: " + str(len(train_dataset)) + "\n")
info.write("\n")

if (DEVICE.type == 'cuda'):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
else:
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

inputs_valid, targets_valid = valid_dataset[:]
inputs_valid = inputs_valid.to(DEVICE)
targets_valid = targets_valid.to(DEVICE)

# Force flash attention to be used (trying it out)
#with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):

model = architectures.MLPBlock(**config['model']).to(DEVICE)
info.write("Total model params: " + str(sum(p.numel() for p in model.parameters())) + "\n")
info.write("Trainable params: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + "\n")
info.write("\n")

optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer'].get('lr', 1.0e-4), weight_decay=config['optimizer'].get('weight_decay', 0.0))

loss_fn = nn.MSELoss(reduction='mean')

training_loss = []
validation_loss = []
min_valid_loss = 1000.0
save_epoch = 0
for epoch in tqdm(range(1, epochs+1)):
    model.train()
    for batch, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        pred = model(inputs)

        loss = loss_fn(pred, targets)

        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value) # gradient clipping
        optimizer.step()

        train_loss_number = loss.detach().to('cpu').item()
        training_loss.append(train_loss_number)
    
    model.eval()
    with torch.no_grad():
        pred_valid = model(inputs_valid)

        loss_valid = loss_fn(pred_valid, targets_valid)

        loss_valid_number = loss_valid.detach().to('cpu').item()
        validation_loss.append(loss_valid_number)

        if ((loss_valid_number <= min_valid_loss) or (epoch == 1)):
            torch.jit.script(model).save(SAVE_FOLDER+"/best_model_torchscript.pt")
            min_valid_loss = loss_valid_number
            save_epoch = epoch

info.write("Saved at epoch: " + str(save_epoch) + "\n")
info.write("Min validation loss: " + str(min_valid_loss) + "\n")
info.write("\n")

torch.jit.script(model.to('cpu')).save(SAVE_FOLDER+"/final_model_state_torchscript_CPU.pt")

np.save(SAVE_FOLDER+"/training_loss.npy", np.array(training_loss))
np.save(SAVE_FOLDER+"/validation_loss.npy", np.array(validation_loss))

best_model = torch.jit.load(SAVE_FOLDER+"/best_model_torchscript.pt").to('cpu')
torch.jit.script(best_model).save(SAVE_FOLDER+"/best_model_torchscript_CPU.pt")

best_model.eval()
with torch.no_grad():
    inputs_test, targets_test = test_dataset[:]
    inputs_test = inputs_test.to('cpu')
    targets_test = targets_test.to('cpu')

    pred_test = best_model(inputs_test)

    loss_test = loss_fn(pred_test, targets_test)
    loss_test_number = loss_test.item()

info.write("Test loss: " + str(loss_test_number) + "\n")
info.close()

test_predictions = {
    'inputs': inputs_test.detach().numpy(),
    'targets': targets_test.detach().numpy(),
    'predictions': pred_test.detach().numpy()
}
np.savez(SAVE_FOLDER+"/test_predictions.npz", **test_predictions)

plotting.plot_loss(training_loss, validation_loss, SAVE_FOLDER+"/loss.png", loss_prefix="MSE")