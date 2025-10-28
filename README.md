# BoostCD

## 1. Setup
Start by cloning the repository:
```bash
git clone https://github.com/epfl-dlab/BoostCD.git
```

We recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment as follows:
```bash
conda env create -f environment.yml
```
This command also installs all the necessary packages.

## 2. Downloading data and models
The data is available on [huggingface](https://huggingface.co/datasets/msakota/boostie) and can be loaded with
```bash
from datasets import load_dataset
dataset = load_dataset("msakota/boostie")
```

## 3. Usage
### Training
To train a model from scratch on the desired data, run:
```bash
# specify a directory where training data is located
RUN_NAME="train_boostie_base_fe"
python src/genie/run_train.py run_name=$RUN_NAME +experiment/train=boostie_base_fe
```
### Inference
To run inference on a trained model:

```bash
CHECKPOINT_PATH="./models/boostie_base_fe.ckpt" # specify path to the trained model
RUN_NAME="inference_boostie_base_fe"
python src/genie/run_inference.py run_name=$RUN_NAME checkpoint_path=$CHECKPOINT_PATH +experiment/inference=boostie_base_fe
```

### Evaluation

To compute the micro and macro performance, as well as the performance bucketed by relation frequency and number of target triplets, you only need the run's WandB path and to execute:

```
python src/genie/run_process_predictions.py +experiment/process_predictions=complete_boostie wandb_run_path=$WANDB_PATH
```

### Citation
If you found our resources useful, please consider citing our work.
```
@misc{šakota2025combiningconstrainedunconstraineddecoding,
      title={Combining Constrained and Unconstrained Decoding via Boosting: BoostCD and Its Application to Information Extraction}, 
      author={Marija Šakota and Robert West},
      year={2025},
      eprint={2506.14901},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.14901}, 
}
```
