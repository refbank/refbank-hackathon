import argparse, ast, warnings, re, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
LETTER_RE = re.compile(r"\b([A-L])\b")

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

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Moondream2
    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    df = pd.read_csv(opt.data_path)
    print(f"Original dataset has {len(df)} rows")
    df = df.head(50)
    print(f"Testing with {len(df)} rows")

    grid_img = Image.open(opt.image_path).convert("RGB")

    rows_out = []
    successful_predictions = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = row["message_history_trunc"] if opt.history_type == "yoked" else row["messages"]
            conv_text = _to_chat_str(_flatten(raw)).strip()
            if not conv_text:
                raise ValueError("empty conv")

            # Create question for Moondream2
            question = (
                f"Look at the tangram pieces labeled A through L in this image. "
                f"Based on this conversation, which letter corresponds to the described shape? "
                f"Answer with just the letter.\n\nConversation:\n{conv_text}"
            )
            
            # Generate response with Moondream2
            with torch.no_grad():
                response = model.answer_question(grid_img, question, tokenizer)
            pred_letter = extract_letter(response)
            valid = pred_letter in "ABCDEFGHIJKL"

            rows_out.append({
                "trial_id": row["trial_id"],
                "model_choice_raw": response.strip(),
                "model_choice": pred_letter,
                "target": row["target"],
                "correct": valid and pred_letter == row["target"],
                "valid_format": valid
            })

            successful_predictions += 1

            if idx < 6:
                print(f"\nTrial {idx} id={row['trial_id']}  tgt={row['target']}  raw='{response.strip()}'  → {pred_letter}")

        except Exception as err:
            print(f"trial {idx} (id={row.get('trial_id','?')}): {err}")
            rows_out.append({
                "trial_id": row.get("trial_id", f"unknown_{idx}"),
                "model_choice_raw": f"ERROR: {str(err)}",
                "model_choice": "?",
                "target": row.get("target", "?"),
                "correct": False,
                "valid_format": False
            })

    # Save results
    out_csv = f"model_choices-moondream2-{opt.experiment_name}-{opt.history_type}.csv"
    
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path", default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())