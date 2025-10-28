from abc import ABC
import hydra
import numpy as np


class Scheduler(ABC):
    def __init__(self, max_transformations, instructions_path, prompter, openai_api, seed):
        self.max_transformations = max_transformations
        self.instructions_path = instructions_path
        self.prompter = prompter
        self.openai_api = openai_api
        self.seed = seed
        np.random.seed(seed)

    def predict_datamodule(self, dataloader, path_to_output_file):
        raise NotImplementedError("Not implemented")