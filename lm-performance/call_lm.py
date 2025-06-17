"""
Call a VLM to
"""

import torch
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from pyprojroot import here
from PIL import Image
from argparse import ArgumentParser

def get_image_token(model_name):
    if "Qwen" in model_name:
        return "<image>"
    elif "gemma" in model_name:
        return "<start_of_image>"
    elif "llama" in model_name:
        return "<|image|>"
    else:
        raise ValueError(f"Model {model_name} not supported")

def extract_answer(response):
    """
    Extract the answer from the response.
    """
    try:
        return response.outputs[0].text.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError:
        return ""


def compute_accuracy(model_choices, labels):
    """
    Compute the accuracy of the model choices.
    """
    return np.mean(np.array(model_choices) == np.array(labels))


def get_user_message(messages):
    """
    Get the user message from a list of messages.
    """
    if not isinstance(messages, list):
        return ""

    user_message = ""
    for message in messages:
        user_message += f"{message['role']}: {message['text']}\n"

    return user_message


def preprocess_messages(df_trials, df_messages):

    df_messages = df_messages.sort_values(["trial_id", "message_number"])
    df_messages["info"] = df_messages.apply(
        lambda x: {
            "text": x["text"],
            "player_id": x["player_id"],
            "role": x["role"],
            "message_number": x["message_number"],
        },
        axis=1,
    )
    df_message_lists = (
        df_messages.groupby("trial_id")["info"]
        .apply(list)
        .reset_index()
        .rename(columns={"info": "messages"})
    )

    df_trials = df_trials.sort_values(["game_id", "rep_num", "trial_num"])
    df_trials = df_trials.merge(df_message_lists, on="trial_id", how="left")

    df_trials["user_message"] = df_trials["messages"].apply(get_user_message)
    df_trials["label"] = df_trials["target"]
    df_trials = df_trials[["user_message", "label"]]

    df_trials = df_trials[df_trials["user_message"] != ""]

    return df_trials


SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.

Think step by step before your answer, with your reasoning contained in <think></think> tags. Then respond with your answer in <answer></answer> tags.
"""

def main(args):

    df_messages = pd.read_csv(here(f"harmonized_data/{args.experiment_name}/messages.csv"))
    df_trials = pd.read_csv(here(f"harmonized_data/{args.experiment_name}/trials.csv"))
    grid_image = Image.open(here("lm-performance/compiled_grid.png"))

    df_trials = preprocess_messages(df_trials, df_messages)
    df_trials = df_trials.sample(args.n_trials)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        max_model_len=8192,
        max_num_seqs=5,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        dtype=torch.bfloat16,
    )

    guided_decoding_params = GuidedDecodingParams(
        choice=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    )

    if args.method == "direct":

        system_prompt = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.

        Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose.
        """

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            guided_decoding=guided_decoding_params,
        )


    elif args.method == "cot":

        system_prompt = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.

        Think step by step before your answer, with your reasoning contained in <think></think> tags. Then respond with your answer in <answer></answer> tags.
        """

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
        )
    else:
        raise ValueError(f"Method {args.method} not supported")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = []
    for user_message in df_trials["user_message"]:
        chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": get_image_token(args.model) + user_message}
            ],
        prompt = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )[0]
        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {"image": grid_image},
        })

    responses = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    print(f"responses: {responses}")
    model_choices = [extract_answer(response) for response in responses]
    df_trials["model_choice"] = model_choices

    model_name = args.model.replace("/", "--")
    df_trials.to_csv(here(f"lm-performance/results/model_choices-{model_name}-{args.experiment_name}-{args.method}.csv"), index=False)

    accuracy = compute_accuracy(model_choices, df_trials["label"])
    print(f"Model accuracy: {accuracy}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--experiment_name", type=str, default="hawkins2020_characterizing_cued")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--method", type=str, default="cot")
    args = parser.parse_args()

    main(args)
