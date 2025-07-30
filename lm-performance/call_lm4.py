# call_llava.py  ── LLaVA 1.6 replacement for your previous Idefics script
import argparse, ast, warnings, json, re, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration

warnings.filterwarnings("ignore", category=UserWarning)   # silence HF chatter
LETTER_RE = re.compile(r"\b([A-L])\b")                    # A–L extractor


# ---------- helpers ----------------------------------------------------
def _flatten(obj):
    """Recursively collect dicts inside lists/strings → list[dict]"""
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    if isinstance(obj, str):
        if not obj.strip() or obj.strip() == "[]":
            return []
        try:
            obj = ast.literal_eval(obj.replace("nan", "''"))
        except (ValueError, SyntaxError):
            return []
    if isinstance(obj, list) and not obj:
        return []
    out, stack = [], [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, dict):
            out.append(x)
    return out[::-1]


def _to_chat_str(dicts):
    return "\n".join(f"{d.get('role', '?')}: {d.get('text','')}" for d in dicts)


def extract_letter(text: str) -> str:
    m = LETTER_RE.search(text.upper().strip())
    return m.group(1) if m else "?"


# ---------- main -------------------------------------------------------
def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16

    processor = LlavaProcessor.from_pretrained(opt.model)
    model = LlavaForConditionalGeneration.from_pretrained(
        opt.model, device_map="auto", torch_dtype=dtype
    ).eval()

    df = pd.read_csv(opt.data_path)

    # --- quick test slice ------------------------------------------------
    print(f"Original dataset has {len(df)} rows")
    df = df.head(500)      # adjust or remove this line as desired
    print(f"Testing with {len(df)} rows")

    grid_img = Image.open(opt.image_path).convert("RGB")

    SYSTEM_PROMPT = (
        "You are shown a conversation between a describer and matcher that refers to an image "
        "grid of tangram puzzle pieces labelled A–L. "
        "Based on **both** the conversation and the image, output **one capital letter (A–L)** "
        "that matches the describer’s target. No explanation."
    )

    rows_out = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = row["message_history_trunc"] if opt.history_type == "yoked" else row["messages"]
            conv_text = _to_chat_str(_flatten(raw)).strip()
            if not conv_text:
                raise ValueError("empty conv")

            prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{conv_text}\n\nAnswer:"
            # LLaVA expects lists for images/text
            inputs = processor(text=[prompt], images=[grid_img], return_tensors="pt")
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    min_new_tokens=1,
                    temperature=0.5,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            # strip the prompt tokens
            pred_raw = processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:],
                                              skip_special_tokens=True)[0]
            pred_letter = extract_letter(pred_raw)
            valid = pred_letter in "ABCDEFGHIJKL"

            rows_out.append({
                "trial_id": row["trial_id"],
                "model_choice_raw": pred_raw,
                "model_choice": pred_letter,
                "target": row["target"],
                "correct": valid and pred_letter == row["target"],
                "valid_format": valid
            })

            if idx < 6:  # preview first few
                print(f"\nTrial {idx} id={row['trial_id']}  tgt={row['target']}  raw='{pred_raw}'  → {pred_letter}")

        except Exception as err:
            print(f"trial {idx} (id={row.get('trial_id','?')}): {err}")

    # ---------- write + stats -------------------------------------------
    out_csv = (
        f"model_choices-{opt.model.replace('/','--')}-"
        f"{opt.experiment_name}-llava-{opt.history_type}.csv"
    )
    res_df = pd.DataFrame(rows_out)
    res_df.to_csv(out_csv, index=False)

    total = len(res_df)
    correct = res_df["correct"].sum()
    valid   = res_df["valid_format"].sum()
    print(f"\nsaved → {out_csv}   ({total} rows)")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"Valid format: {valid}/{total} = {valid/total*100:.2f}%")
    print(f"Random baseline: {100/12:.2f}%")

    if valid:
        from collections import Counter
        dist = Counter(res_df.loc[res_df.valid_format, "model_choice"])
        print("\nPrediction distribution (valid only):")
        for l in "ABCDEFGHIJKL":
            if dist[l]:
                print(f"  {l}: {dist[l]} ({dist[l]/valid*100:.1f}%)")
        if max(dist.values()) > valid * 0.3:
            mc = dist.most_common(1)[0]
            print(f"\n⚠️  Bias warning – '{mc[0]}' appears {mc[1]/valid*100:.1f}% of valid predictions")


# ---------- CLI --------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llava-hf/llava-1.6-7b-hf",
                   help="Any open LLaVA checkpoint on HF hub")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path", default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())
