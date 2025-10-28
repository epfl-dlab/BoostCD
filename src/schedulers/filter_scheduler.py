from src.schedulers import RandomScheduler
import numpy as np


class FilterRandomScheduler(RandomScheduler):
    def __init__(self, max_transformations, instructions_path, prompter, openai_api, seed=0, filter=None, aligner=None):
        super().__init__(max_transformations, instructions_path, prompter, openai_api, seed)
        self.filter = filter
        self.aligner = aligner

    def predict(self, batch):
        if not isinstance(batch, list):
            batch = [batch]

        for sample in batch:
            curr_sample = sample.copy()
            sample["prompt"] = []
            sample["model_completions"] = []
            sample["instructions"] = []
            sample["instruction_types"] = []
            sample["triplets_old"] = sample["triplets"].copy()
            sample["text_old"] = sample["text"]
            if np.random.rand() > 0.5:
                curr_sample = self.filter.process_sample(curr_sample)
                sample["initial_transformation"] = "filter"
            else:
                curr_sample = self.aligner.align_sample(curr_sample)
                sample["initial_transformation"] = "align"
            sample["triplets"] = curr_sample["triplets"].copy()
            sample["text"] = curr_sample["text"]
            sample["target"] = curr_sample["target"]
            n_transformations = np.random.choice(
                self.max_transformations) + 1  # randomly get number of transformations to be performed (zero not included)
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
                curr_sample["text"] = compl
                if compl == "":
                    break
                if instruction_type.startswith("context") or instruction_type.startswith("style"):
                    curr_sample = self.aligner.align_sample(curr_sample)
                    sample["instructions"].append("align")
                    sample["instruction_types"].append("align")
                    sample["model_completions"].append(curr_sample["text"])
                    if curr_sample["text"] == "":
                        break
        if hasattr(self, "outputs"):
            self.outputs.extend(batch)

