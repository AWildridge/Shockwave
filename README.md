# Shockwave

Our project for the [HDR ML Challenge](https://www.codabench.org/competitions/2626/) for anomalous gravitational wave detection.

## YAML Configs
The current (this is first version) `Train.py` script expects a YAML config file, giving model, dataset, optimizer, and loss hyperparameters. The best teacher is an example, so here's my initial one:

```
model:
  d_input: 200
  hidden_layers: [400, 400]
  d_output: 200
  dropout: 0.0

dataset:
  location: "/path/to/folder/where/npy/files/live"
  batch_size: 200

optimizer:
  epochs: 15
  lr: 1.0e-4
  weight_decay: 0.0

loss:
  type: "MSE"
```
The barest information you need in `config.yaml` is:
- All hyperparameters needed to instantiate your model
- The location of a folder containing your dataset files

Everything else has default values in the script, but it'd be good to include them as well, since it also serves as a form of documentation for your trainings. I even include some things (like the loss "type") that aren't used by the script, and are there for me to look at when I'm browsing through old trainings.