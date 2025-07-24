
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from argparse import ArgumentParser
from functools import partial
from ast import literal_eval
import os
from tqdm import tqdm


def get_image_token(model_name):
    if "gemma" in model_name:
        return "<start_of_image>"
    elif "llava" in model_name:
        return "<image>"
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
        return text.strip()[-1].upper()


def compute_accuracy(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def get_user_message(messages):
    if not isinstance(messages, list):
        return ""
    return "\n".join(f"{m['role']}: {m['text']}" for m in messages)


def preprocess_messages(row, history_type):
    messages = []

    if history_type != "none":
        hist = row.get("message_history_trunc", "[]")
        target = row.get("target_history_trunc", "[]")
        hist = literal_eval(hist.replace("nan", "''")) if isinstance(hist, str) else []
        target = literal_eval(target) if isinstance(target, str) else []

        for h, t in zip(hist, target):
            messages.append({"role": "user", "content": get_user_message(h)})
            messages.append({"role": "assistant", "content": t})

    trial = row.get("messages", "")
    if isinstance(trial, str):
        trial = literal_eval(trial.replace("nan", "''"))
    else:
        trial = []

    messages.append({"role": "user", "content": get_user_message(trial)})
    return messages


def main(args):
    df = pd.read_csv("trials_with_history.csv")
    if args.n_trials:
        df = df.head(args.n_trials)

    # Load image
    image_token = get_image_token(args.model)
    grid_image = Image.open("compiled_grid.png")

    # Shuffle history
    if args.history_type == "shuffled":
        perm = np.random.permutation(len(df))
        df["message_history_trunc"] = df["message_history_trunc"].iloc[perm].reset_index(drop=True)
        df["target_history_trunc"] = df["target_history_trunc"].iloc[perm].reset_index(drop=True)

    df["chat_prompt"] = df.apply(partial(preprocess_messages, history_type=args.history_type), axis=1)
    print("Sample prompt:\n", df["chat_prompt"].iloc[0])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

    if args.method == "direct":
        system_prompt = (
            "You are given a chat between a describer and matcher about a mystery object. "
            "They are trying to identify which labeled tangram (A to L) is being referred to.\n"
            "Refer to the image.\n\n"
            "Answer with a single letter only.\n\n"
        )
    else:
        system_prompt = (
            "You are shown a dialogue between a describer and matcher trying to guess a labeled image (A to L).\n"
            "Think carefully step-by-step in <think>...</think>, and then give the answer in <answer>...</answer>.\n"
        )

    model_choices = []
    for chat_prompt in tqdm(df["chat_prompt"]):
        chat = [{"role": "system", "content": system_prompt + image_token}] + chat_prompt
        full_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_choices.append(extract_answer(decoded))

    df["model_choice"] = model_choices
    acc = compute_accuracy(model_choices, df["target"])
    print(f"Accuracy: {acc:.3f}")

    os.makedirs("results", exist_ok=True)
    fname = f"results/model_choices-{args.model.replace('/', '--')}-{args.experiment_name}-{args.method}-history-{args.history_type}.csv"
    df.to_csv(fname, index=False)
    print("Saved to", fname)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--experiment_name", type=str, default="hawkins2020_characterizing_cued")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--method", type=str, choices=["direct", "cot"], default="direct")
    parser.add_argument("--history_type", choices=["yoked", "shuffled", "none"], default="yoked")
    args = parser.parse_args()
    main(args)
