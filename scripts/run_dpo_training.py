import os
import sys
import argparse
import json
from trl import DPOTrainer
from trl import DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
sys.path.append("./")
from src import utils
from src.genie.models import GenIEFlanT5PL
#log = utils.get_pylogger(__name__)
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a Language model")
    parser.add_argument("--data_path", required=True, type=str, help="Path to the ranking data")
    parser.add_argument("--val_path", default=None, type=str, help="Path to validation data")
    parser.add_argument("--model_name", required=True, type=str, help="Path to the model")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--save_pl", action="store_true")
    parser.add_argument("--save_huggingface", action="store_true")
    parser.add_argument("--read_from_pl", action="store_true")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default= 1e-5)
    parser.add_argument("--run_name", type=str, default="boostie_base_fe_dpo")
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project="CurIE", name=args.run_name)

    # load dataset
    with open(args.data_path, 'r') as f:
        datasamples = json.load(f)

    data = Dataset.from_dict(datasamples)

    val_data = None
    if args.val_path:
        with open(args.val_path, 'r') as f:
            datasamples = json.load(f)

        val_data = Dataset.from_dict(datasamples)

    # set training arguments
    training_args = DPOConfig(
        beta=args.beta,
        output_dir=args.save_dir,
        max_prompt_length=512,
        max_target_length=256,
        remove_unused_columns=False,
        per_device_train_batch_size=args.bs,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        learning_rate= 1e-5,
    )

    DATA_DIR = "./data"

    if args.read_from_pl:
        model_pl = GenIEFlanT5PL.load_from_checkpoint(checkpoint_path=os.path.join(DATA_DIR, "model/ie_models/", f"{args.model_name}.ckpt"))
        model_ref_pl = GenIEFlanT5PL.load_from_checkpoint(checkpoint_path=os.path.join(DATA_DIR, "model/ie_models/", f"{args.model_name}.ckpt"))
        model = model_pl.model
        model_ref = model_ref_pl.model
        tokenizer = model_pl.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model_ref = AutoModelForCausalLM.from_pretrained(args.model_name)

    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token

    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    from transformers import TrainerCallback

    class WandBLogValidationOutputsCallback(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer  # Store trainer instance

        def on_evaluate(self, args, state, control, **kwargs):
            # Get the current step
            step = state.global_step

            # Get validation dataset and select a few examples
            val_dataset = self.trainer.eval_dataset
            example_texts = [val_dataset[i]["prompt"] for i in range(min(3, len(val_dataset)))]  # First 3 examples

            # Tokenize and generate responses
            inputs = self.trainer.tokenizer(example_texts, return_tensors="pt", padding=True,
                                            truncation=True).input_ids.to(self.trainer.model.device)
            responses = self.trainer.model.generate(inputs, max_length=100)
            decoded_responses = self.trainer.tokenizer.batch_decode(responses, skip_special_tokens=True)

            # Create a WandB Table to log the Q&A pairs for the current evaluation step
            table = wandb.Table(columns=["Step", "Question", "Answer"])
            for q, a in zip(example_texts, decoded_responses):
                table.add_data(step, q, a)  # Add step, question, and answer to the table

            # Log the table to WandB with the current step as part of the log
            wandb.log({f"val_examples_step_{step}": table})

    wandb_callback = WandBLogValidationOutputsCallback(trainer)
    trainer.add_callback(wandb_callback)

    trainer.train()

    # saving model
    if args.save_huggingface:
        trainer.save_model(args.save_dir)


if __name__ == "__main__":
    main()