"""
Call a VLM to
"""

import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pyprojroot import here


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


if __name__ == "__main__":

    # TODO: make these arguments that you can change at some point
    MODEL = "meta-llama/Llama-3.2-3B"
    EXPERIMENT_NAME = "hawkins2020_characterizing_cued"

    df_messages = pd.read_csv(here(f"harmonized_data/{EXPERIMENT_NAME}/messages.csv"))
    df_trials = pd.read_csv(here(f"harmonized_data/{EXPERIMENT_NAME}/trials.csv"))

    df_trials = preprocess_messages(df_trials, df_messages)

    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        max_model_len=4096,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        gpu_memory_utilization=0.95,
    )

    guided_decoding_params = GuidedDecodingParams(
        choice=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2,
    )

    responses = llm.generate(
        prompts=df_trials["user_message"].tolist(),
    )
    model_choices = [response.outputs[0].text for response in responses]

    accuracy = compute_accuracy(model_choices, df_trials["label"])
    print(f"Model accuracy: {accuracy}")
