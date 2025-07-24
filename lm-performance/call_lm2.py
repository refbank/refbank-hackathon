import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from pyprojroot import here
from PIL import Image
from argparse import ArgumentParser
from functools import partial
from ast import literal_eval
import os
from tqdm import tqdm


def extract_answer(text):
    try:
        return text.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError:
        return text.strip()

def compute_accuracy(model_choices, labels):
    return np.mean(np.array(model_choices) == np.array(labels))

def get_user_message(messages):
    if not isinstance(messages, list):
        return ""
    user_message = ""
    for message in messages:
        user_message += f"{message['role']}: {message['text']}\n"
    return user_message

def preprocess_messages(row, history_type):
    chat_messages = ""
    if history_type != "none":
        message_history = literal_eval(row["message_history_trunc"].replace("nan", "''")) if isinstance(row["message_history_trunc"], str) else []
        target_history = literal_eval(row["target_history_trunc"]) if isinstance(row["target_history_trunc"], str) else []

        for messages, target in zip(message_history, target_history):
            user_message = get_user_message(messages)
            chat_messages += f"user: {user_message}\nassistant: {target}\n"

    this_trial_messages = row["messages"]
    if not isinstance(this_trial_messages, str):
        chat_messages += "user: describer: \n"
    else:
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages += "user: " + get_user_message(this_trial_messages)

    return chat_messages

def main(args):
    df = pd.read_csv(here("lm-performance/trials_with_history.csv"))
    if args.n_trials is not None:
        df = df.head(args.n_trials)

    image = Image.open(here("lm-performance/compiled_grid.png")).convert("RGB")

    if args.history_type == "shuffled":
        print("Shuffling histories")
        perm = np.random.permutation(len(df))
        df["message_history_trunc"] = df["message_history_trunc"].iloc[perm].reset_index(drop=True)
        df["target_history_trunc"] = df["target_history_trunc"].iloc[perm].reset_index(drop=True)

    df["chat_prompt"] = df.apply(partial(preprocess_messages, history_type=args.history_type), axis=1)
    print("Example chat prompt:\n", df["chat_prompt"].iloc[0])

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)

    model_choices = []
    for prompt in tqdm(df["chat_prompt"]):
        full_prompt = (
            "You are shown a conversation between a describer and matcher trying to identify an image among labeled options (A to L). "
            "Based on the conversation and the image, guess which tangram (labeled A to L) is being described.\n"
            "Answer with your best guess in the format <answer>A</answer>\n\n"
            + prompt
        )

        inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        answer = extract_answer(decoded)
        model_choices.append(answer)

    df["model_choice"] = model_choices
    model_name = args.model.replace("/", "--")
    out_path = here(f"lm-performance/results/model_choices-{model_name}-{args.experiment_name}-{args.history_type}.csv")
    df[["trial_id", "stage_num", "rep_num", "trial_num", "chat_prompt", "model_choice", "target"]].to_csv(out_path, index=False)

    acc = compute_accuracy(model_choices, df["target"])
    print(f"Model accuracy: {acc:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceM4/idefics-9b-instruct")
    parser.add_argument("--experiment_name", type=str, default="hawkins2020_characterizing_cued")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--history_type", type=str, default="yoked", choices=["yoked", "shuffled", "none"])
    args = parser.parse_args()

    main(args)
