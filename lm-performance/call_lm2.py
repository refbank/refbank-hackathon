import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyprojroot import here
from PIL import Image
from argparse import ArgumentParser
from functools import partial
from ast import literal_eval
import os
from tqdm import tqdm


def get_image_token(model_name):
    if "gemma" in model_name:
        return "<start_of_image>"
    elif "llama" in model_name:
        return "<|image|>"
    elif "idefics" in model_name:
        return "<image>"
    elif "Qwen" in model_name:
        return "<|vision_bos|><|image_pad|><|vision_eos|>"
    else:
        raise ValueError(f"Model {model_name} not supported")


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

    grid_image = Image.open(here("lm-performance/compiled_grid.png"))

    if args.history_type == "shuffled":
        print("shuffling histories")
        perm = np.random.permutation(len(df_with_history))
        df_with_history["message_history_trunc"] = df_with_history["message_history_trunc"].iloc[perm]
        df_with_history["target_history_trunc"] = df_with_history["target_history_trunc"].iloc[perm]

    df_with_history["chat_prompt"] = df_with_history.apply(
        partial(preprocess_messages, history_type=args.history_type), axis=1
    )

    print(f"example chat prompt: {df_with_history['chat_prompt'].sample(1).iloc[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.method == "direct":
        system_prompt = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.

Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose.
"""
    elif args.method == "cot":
        system_prompt = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.

Think step by step before your answer, with your reasoning contained in <think></think> tags. Then respond with your answer in <answer></answer> tags.
"""
    else:
        raise ValueError(f"Method {args.method} not supported")

    model_choices = []
    for chat_prompt in tqdm(df_with_history["chat_prompt"]):
        chat = (
            [
                {"role": "system", "content": system_prompt + get_image_token(args.model)},
                *chat_prompt,
            ]
        )
        prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=32, temperature=0.0)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = extract_answer(decoded)
        model_choices.append(answer)

    df_with_history["model_choice"] = model_choices

    df_with_history = df_with_history[["trial_id", "stage_num", "rep_num", "trial_num", "chat_prompt", "model_choice", "target"]]
    model_name = args.model.replace("/", "--")
    df_with_history.to_csv(
        here(f"lm-performance/results/model_choices-{model_name}-{args.experiment_name}-{args.method}-history-{args.history_type}.csv"),
        index=False,
    )

    accuracy = compute_accuracy(model_choices, df_with_history["target"])
    print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--experiment_name", type=str, default="hawkins2020_characterizing_cued")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--method", type=str, default="direct")
    parser.add_argument("--history_type", type=str, default="yoked", choices=["yoked", "shuffled", "none"])
    args = parser.parse_args()

    main(args)
