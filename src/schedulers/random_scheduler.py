from src.schedulers.abstract import Scheduler
import numpy as np
import json
from tqdm import tqdm


class RandomScheduler(Scheduler):
    def __init__(self, max_transformations, instructions_path, prompter, openai_api, seed=0):
        super().__init__(max_transformations, instructions_path, prompter, openai_api, seed)
        self.instructions = None
        self.load_instructions(instructions_path)
        self.total_num_instructions = len(self.instructions)

    def load_instructions(self, instructions_path):
        with open(instructions_path, 'r') as f:
            self.instructions = json.load(f)

    def predict(self, batch):
        # if batch is not a list wrap it in a list
        if not isinstance(batch, list):
            batch = [batch]

        for sample in batch:
            n_transformations = np.random.choice(self.max_transformations) + 1 # randomly get number of transformations to be performed (zero not included)
            completion_steps = []
            curr_sample = sample.copy()
            sample["prompt"] = []
            sample["model_completions"] = []
            sample["instructions"] = []
            sample["instruction_types"] = []
            instructions_set = self.instructions.copy()

            for i in range(n_transformations):
                random_index = np.random.choice(len(instructions_set)) # randomly get the next transformation to be performed
                instruction_type = list(instructions_set.keys())[random_index]
                system_content = instructions_set[instruction_type]

                instructions_set.pop(instruction_type)
                if instruction_type == "context_positive":
                    instructions_set.pop("context_negative")
                if instruction_type == "context_negative":
                    instructions_set.pop("context_positive")

                prompt = self.prompter(curr_sample, system_content)
                api_key_idx = self.openai_api._choose_next_api_key()
                completions = self.openai_api._get_text_completions([prompt], api_key_idx)
                cleaned_completion = self.openai_api.clean_completions(completions)
                sample["prompt"].append(prompt)
                try:
                    compl = cleaned_completion[0][0]
                except:
                    compl = ""

                sample["model_completions"].append(compl)
                sample["instructions"].append(system_content)
                sample["instruction_types"].append(instruction_type)
                completion_steps.append(curr_sample)
                curr_sample["text"] = compl
                if compl == "":
                    break

        if hasattr(self, "outputs"):
            self.outputs.extend(batch)

    def predict_datamodule(self, dataloader, output_file):
        self.outputs = []

        for batch in tqdm(dataloader):
            self.predict(batch)

        # n_workers = self.n_workers_per_key * len(self.api_keys)
        # with ThreadPoolExecutor(max_workers=n_workers) as executor:
        #     executor.map(self.predict, dataloader)

            if not self.openai_api.dry_run and output_file is not None:
                self.openai_api.write_outputs(output_file, self.outputs)
            self.outputs = []

        del self.outputs

