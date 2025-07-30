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
    df = df.head(50)      # adjust or remove this line as desired
    print(f"Testing with {len(df)} rows")

    grid_img = Image.open(opt.image_path).convert("RGB")

    # Different prompt approach - avoid bias
    SYSTEM_PROMPT = (
        "Look at the tangram puzzle pieces in this image. Each piece is labeled with a letter from A to L. "
        "Based on the conversation below, identify which letter corresponds to the shape being described. "
        "Reply with only the single letter."
    )

    rows_out = []
    successful_predictions = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = row["message_history_trunc"] if opt.history_type == "yoked" else row["messages"]
            conv_text = _to_chat_str(_flatten(raw)).strip()
            if not conv_text:
                raise ValueError("empty conv")

            # Simpler prompt format that works better with older transformers versions
            full_prompt = f"USER: <image>\n{SYSTEM_PROMPT}\n\nConversation:\n{conv_text}\n\nASSISTANT:"
            
            # Process inputs with simplified approach
            inputs = processor(
                text=full_prompt, 
                images=grid_img, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            with torch.no_grad():
                # Use greedy decoding to reduce bias
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    min_new_tokens=1,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            # Strip the prompt tokens to get only the generated response
            new_tokens = generated_ids[:, inputs['input_ids'].shape[1]:]
            pred_raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
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

            successful_predictions += 1

            if idx < 6:  # preview first few
                print(f"\nTrial {idx} id={row['trial_id']}  tgt={row['target']}  raw='{pred_raw}'  → {pred_letter}")

        except Exception as err:
            print(f"trial {idx} (id={row.get('trial_id','?')}): {err}")
            # Add a failed row to maintain consistent DataFrame structure
            rows_out.append({
                "trial_id": row.get("trial_id", f"unknown_{idx}"),
                "model_choice_raw": f"ERROR: {str(err)}",
                "model_choice": "?",
                "target": row.get("target", "?"),
                "correct": False,
                "valid_format": False
            })

    # ---------- write + stats -------------------------------------------
    out_csv = (
        f"model_choices-{opt.model.replace('/','--')}-"
        f"{opt.experiment_name}-llava-{opt.history_type}.csv"
    )
    
    # Create DataFrame even if all predictions failed
    if not rows_out:
        print("No predictions were made!")
        return
        
    res_df = pd.DataFrame(rows_out)
    res_df.to_csv(out_csv, index=False)

    total = len(res_df)
    correct = res_df["correct"].sum()
    valid = res_df["valid_format"].sum()
    
    print(f"\nSaved → {out_csv} ({total} rows)")
    print(f"Successful predictions: {successful_predictions}/{total}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"Valid format: {valid}/{total} = {valid/total*100:.2f}%")
    print(f"Random baseline: {100/12:.2f}%")

    if valid > 0:
        from collections import Counter
        valid_predictions = res_df[res_df["valid_format"]]
        dist = Counter(valid_predictions["model_choice"])
        print("\nPrediction distribution (valid only):")
        for l in "ABCDEFGHIJKL":
            if dist[l] > 0:
                print(f"  {l}: {dist[l]} ({dist[l]/valid*100:.1f}%)")
        
        if valid > 0 and max(dist.values()) > valid * 0.3:
            mc = dist.most_common(1)[0]
            print(f"\n⚠️  Bias warning – '{mc[0]}' appears {mc[1]/valid*100:.1f}% of valid predictions")
    else:
        print("\nNo valid predictions to analyze distribution")


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