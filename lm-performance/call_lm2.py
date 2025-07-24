import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import pandas as pd
from pyprojroot import here
from argparse import ArgumentParser
from functools import partial
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import os


def get_user_message(messages):
    if not isinstance(messages, list):
        return ""
    user_message = ""
    for message in messages:
        user_message += f"{message['role']}: {message['text']}\n"
    return user_message


def preprocess_messages(row, history_type):
    chat_messages = []
    if history_type != "none":
        message_history_trunc = row["message_history_trunc"]
        if not isinstance(message_history_trunc, str):
            message_history = []
        else:
            message_history = literal_eval(message_history_trunc.replace("nan", "''"))

        target_history = literal_eval(row["target_history_trunc"])
        for messages, target in zip(message_history, target_history):
            user_message = get_user_message(messages)
            chat_messages.append({"role": "user", "content": user_message})
            chat_messages.append({"role": "assistant", "content": target})

    this_trial_messages = row["messages"]
    if not isinstance(this_trial_messages, str):
        chat_messages.append({"role": "user", "content": "describer: \n"})
    else:
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages.append({"role": "user", "content": get_user_message(this_trial_messages)})

    return chat_messages


def main(args):
    df_with_history = pd.read_csv(here("lm-performance/trials_with_history.csv"))
    if args.n_trials is not None:
        df_with_history = df_with_history.head(args.n_trials)

    if args.history_type == "shuffled":
        print("shuffling histories")
        perm = np.random.permutation(len(df_with_history))
        df_with_history["message_history_trunc"] = df_with_history["message_history_trunc"].iloc[perm]
        df_with_history["target_history_trunc"] = df_with_history["target_history_trunc"].iloc[perm]

    df_with_history["chat_prompt"] = df_with_history.apply(
        partial(preprocess_messages, history_type=args.history_type), axis=1
    )

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    image = Image.open(here("lm-performance/compiled_grid.png"))

    def extract_answer(text):
        try:
            return text.split("<answer>")[1].split("</answer>")[0].strip()
        except IndexError:
            return text.strip()

    if args.method == "direct":
        system_prompt = "You will be shown a grid of images. The describer is trying to get the matcher to choose the correct image. Just answer with the image label.\n\n"
    else:
        system_prompt = "You will be shown a grid of images. The describer is trying to get the matcher to choose the correct image. Think step by step in <think> tags, then answer in <answer> tags.\n\n"

    model_choices = []
    for chat_prompt in tqdm(df_with_history["chat_prompt"]):
        full_text = system_prompt + chat_prompt[-1]["content"]
        inputs = processor(text=full_text, images=image, return_tensors="pt").to("cuda", torch.float16)
        outputs = model.generate(**inputs, max_new_tokens=32)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        answer = extract_answer(decoded)
        model_choices.append(answer)

    df_with_history["model_choice"] = model_choices
    df_with_history = df_with_history[["trial_id", "stage_num", "rep_num", "trial_num", "chat_prompt", "model_choice", "target"]]
    model_name = args.model.replace("/", "--")
    df_with_history.to_csv(
        here(f"lm-performance/results/model_choices-{model_name}-{args.experiment_name}-{args.method}-history-{args.history_type}.csv"),
        index=False,
    )

    accuracy = np.mean(np.array(model_choices) == np.array(df_with_history["target"]))
    print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceM4/idefics-9b-instruct")
    parser.add_argument("--experiment_name", type=str, default="hawkins2020_characterizing_cued")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--history_type", type=str, default="yoked", choices=["yoked", "shuffled", "none"])
    args = parser.parse_args()
    main(args)
