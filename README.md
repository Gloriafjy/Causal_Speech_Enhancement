## Installation

First, install Python (recommended with Anaconda).

## Development

Clone this repository and install the dependencies. We recommend using
a fresh Conda environment.

```bash
git clone https://github.com/Gloriafjy/Causal_Speech_Enhancement
cd code
pip install -r requirements_cuda.txt
```

## Train and evaluate

### 1. Data

Run `sh make_valentini.sh` to generate json files.



### 2. Train
Training is simply done by launching the `train.py` script:
This scripts read all the configurations from the `conf/config.yaml` file.

### 3. Logs

Logs are stored by default in the `outputs` folder. 
In the experiment folder you will find the `best.th` serialized model, the training checkpoint `checkpoint.th`,
and well as the log with the metrics `trainer.log`. All metrics are also extracted to the `history.json`
file for easier parsing. Enhancements samples are stored in the `samples` folder (if `noisy_dir` or `noisy_json`
is set in the dataset).


### 4. Evaluate

Evaluating the models can be done by:

```
python -m model.evaluate --model_path=<path to the model> --data_dir=<path to folder containing noisy.json and clean.json>
```
Note that the path given to `--model_path` should be obtained from one of the `best.th` file, not `checkpoint.th`.
