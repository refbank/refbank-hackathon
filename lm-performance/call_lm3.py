# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pyprojroot import here
from PIL import Image
from argparse import ArgumentParser
from functools import partial
from ast import literal_eval
import os
from tqdm import tqdm
import re

def extract_answer(text):
    # Extract a single capital letter A–L using regex
    match = re.search(r"\b([A-L])\b", text.strip())
    return match.group(1) if match else "?"

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
        message_history_trunc = row["message_history_trunc"]
        if not isinstance(message_history_trunc, str):
            message_history = []
        else:
            message_history = literal_eval(message_history_trunc.replace("nan", "''"))

        target_history = literal_eval(row["target_history_trunc"])
        for messages, target in zip(message_history, target_history):
            user_message = get_user_message(messages)
            chat_messages += f"user: {user_message}\nassistant: {target}\n"

    this_trial_messages = row["messages"]
    if not isinstance(this_trial_messages, str):
        chat_messages += "user: describer: \n"
    else:
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages += f"user: {get_user_message(this_trial_messages)}"

    return chat_messages

def main(args):
    df_with_history = pd.read_csv("lm-performance/trials_with_history.csv")
    if args.n_trials is not None:
        df_with_history = df_with_history.head(args.n_trials)

    grid_image = Image.open("lm-performance/compiled_grid.png").convert("RGB")

    if args.history_type == "shuffled":
        print("shuffling histories")
        perm = np.random.permutation(len(df_with_history))
        df_with_history["message_history_trunc"] = df_with_history["message_history_trunc"].iloc[perm].reset_index(drop=True)
        df_with_history["target_history_trunc"] = df_with_history["target_history_trunc"].iloc[perm].reset_index(drop=True)

    df_with_history["chat_prompt"] = df_with_history.apply(
        partial(preprocess_messages, history_type=args.history_type), axis=1
    )

    print(f"example chat prompt: {df_with_history['chat_prompt'].sample(1).iloc[0]}")

    processor = Blip2Processor.from_pretrained(args.model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.model, device_map="auto")

    model_choices = []
    for chat_prompt in tqdm(df_with_history["chat_prompt"]):
        full_prompt = (
            "You are shown a conversation between a describer and matcher trying to identify an image among labeled options (A to L). "
            "Based on the conversation and the image, guess which tangram (labeled A to L) is being described.\n"
            "Answer with a single capital letter from A to L. Do not include any explanation.\n\n"
            + chat_prompt
        )

        inputs = processor(images=grid_image, text=full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        decoded = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = extract_answer(decoded)
        model_choices.append(answer)

    df_with_history["model_choice"] = model_choices

    df_with_history = df_with_history[["trial_id", "stage_num", "rep_num", "trial_num", "chat_prompt", "model_choice", "target"]]
    model_name = args.model.replace("/", "--")
    df_with_history.to_csv(
        f"lm-performance/results/model_choices-{model_name}-{args.experiment_name}-blip2-history-{args.history_type}.csv",
        index=False,
    )

    accuracy = compute_accuracy(model_choices, df_with_history["target"])
    print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "cot"])
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--experiment_name", type=str, default="hawkins2020_characterizing_cued")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--history_type", type=str, default="yoked", choices=["yoked", "shuffled", "none"])
    args = parser.parse_args()

    main(args)
