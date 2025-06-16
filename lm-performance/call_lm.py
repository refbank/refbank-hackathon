"""
Call a VLM to
"""

import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from pyprojroot import here
from PIL import Image

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


if __name__ == "__main__":

    # TODO: make these arguments that you can change at some point
    MODEL = "google/gemma-3-27b-it"
    EXPERIMENT_NAME = "hawkins2020_characterizing_cued"

    df_messages = pd.read_csv(here(f"harmonized_data/{EXPERIMENT_NAME}/messages.csv"))
    df_trials = pd.read_csv(here(f"harmonized_data/{EXPERIMENT_NAME}/trials.csv"))
    grid_image = Image.open(here("lm-performance/compiled_grid.png"))

    df_trials = preprocess_messages(df_trials, df_messages)
    df_trials = df_trials.head(1000)

    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        max_model_len=8192,
        max_num_seqs=5,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        
    )

    guided_decoding_params = GuidedDecodingParams(
        choice=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        # guided_decoding=guided_decoding_params,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    prompts = []
    for user_message in df_trials["user_message"]:
        chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<|image|>" + user_message}
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
    model_choices = [extract_answer(response) for response in responses]
    df_trials["model_choice"] = model_choices

    model_name = MODEL.replace("/", "--")
    df_trials.to_csv(here(f"lm-performance/model_choices-{model_name}-{EXPERIMENT_NAME}.csv"), index=False)

    accuracy = compute_accuracy(model_choices, df_trials["label"])
    print(f"Model accuracy: {accuracy}")
