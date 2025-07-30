import argparse, ast, warnings, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, IdeficsForVisionText2Text
import random

warnings.filterwarnings("ignore", category=UserWarning)   # silence HF chatter

# ---------- helpers ----------------------------------------------------
def _flatten(obj):
    "Recursively collect dicts inside lists/strings → list[dict]"
    if obj is None or (isinstance(obj, float) and pd.isna(obj)):
        return []
    
    if isinstance(obj, str):
        # Handle empty string cases
        if not obj.strip() or obj.strip() == '[]':
            return []
        try:
            # Try to parse the string representation
            obj = ast.literal_eval(obj.replace("nan", "''"))
        except (ValueError, SyntaxError) as e:
            print(f"Failed to parse string with ast.literal_eval: {e}")
            return []
    
    # Handle the case where obj is already a list
    if isinstance(obj, list) and not obj:
        return []
    
    out = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, dict):
            out.append(x)
    return out[::-1]   # keep original order

def _to_chat_str(msg_dicts):
    return "\n".join(f"{d.get('role','?')}: {d.get('text','')}" for d in msg_dicts)

def extract_letter_prediction(text):
    """Extract the most likely letter prediction from model output"""
    import re
    
    # Clean the text first
    text = text.strip().upper()  # Convert to uppercase
    
    # Look for a standalone letter A-L
    match = re.search(r'\b([A-L])\b', text)
    if match:
        return match.group(1)
    
    # Look for any A-L letter
    match = re.search(r'([A-L])', text)
    if match:
        return match.group(1)
    
    # If no valid letter found, return the original text for debugging
    return text

# ---------- main -------------------------------------------------------
def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    proc = AutoProcessor.from_pretrained(opt.model)
    model = IdeficsForVisionText2Text.from_pretrained(
        opt.model, device_map="auto", torch_dtype=dtype
    ).eval()

    df = pd.read_csv(opt.data_path)
    
    # TEST MODE: Use first 50 rows to get better statistics
    print(f"Original dataset has {len(df)} rows")
    df = df.head(500)  # More rows for better pattern detection
    print(f"Testing with {len(df)} rows only")
    
    grid = Image.open(opt.image_path).convert("RGB")

    # Try a completely different approach - avoid any ordering or listing
    system_prompt = (
        "Study and look at the 12 images showing tangram puzzle pieces labeled A,B,C,D,E,F,G,H,I,J,K,L.\n"
        "Read and analyze the conversation describing ONE of the 12 tangram piece.\n"
        "Identify which tangram piece is being described by its label letter (A,B,C,D,E,F,G,H,I,J,K,L).\n\n"
    )

    rows_out = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="trials"):
        try:
            raw = row["message_history_trunc"] if opt.history_type == "yoked" \
                  else row["messages"]
            txt = _to_chat_str(_flatten(raw)).strip()
            if not txt:
                raise ValueError("empty conv")

            # Even simpler prompt - no examples that might bias toward specific letters
            prompt = f"{system_prompt}Conversation:\n{txt}\n\nThe label is:"
            inputs = proc(images=grid, text=prompt, return_tensors="pt")
            
            # Move to device only (no dtype conversion for input tensors)
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)
            
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, 
                    max_new_tokens=3,   
                    min_new_tokens=1,   
                    do_sample=True,     
                    temperature=0.4,    # Lower temperature for more consistent responses
                    top_p=0.9,         # Back to higher top_p
                    pad_token_id=proc.tokenizer.eos_token_id,
                    eos_token_id=proc.tokenizer.eos_token_id,
                )
            
            # Extract only the generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_ids = out_ids[:, input_length:]
            pred_raw = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract clean letter prediction
            pred_letter = extract_letter_prediction(pred_raw)
            
            # Validate it's a single letter A-L
            is_valid = pred_letter in 'ABCDEFGHIJKL' and len(pred_letter) == 1
            
            rows_out.append({
                "trial_id": row["trial_id"], 
                "model_choice_raw": pred_raw,
                "model_choice": pred_letter,
                "target": row["target"],
                "correct": is_valid and pred_letter == row["target"],
                "valid_format": is_valid
            })
            
            # Debug output for first few trials
            if idx < 6:
                print(f"\nTrial {idx} (id={row['trial_id']}):")
                print(f"  Target: {row['target']}")
                print(f"  Raw output: '{pred_raw}'")
                print(f"  Extracted: '{pred_letter}'")
                print(f"  Valid: {is_valid}")

        except Exception as e:
            print(f"trial {idx} (id={row.get('trial_id', '?')}): {e}")

    out_csv = (
        f"model_choices-{opt.model.replace('/','--')}-"
        f"{opt.experiment_name}-idefics-{opt.history_type}.csv"
    )
    
    results_df = pd.DataFrame(rows_out)
    results_df.to_csv(out_csv, index=False)
    
    # Enhanced analysis
    if len(rows_out) > 0:
        correct = sum(row["correct"] for row in rows_out)
        valid_format = sum(row["valid_format"] for row in rows_out)
        total = len(rows_out)
        accuracy = correct / total * 100
        format_accuracy = valid_format / total * 100
        
        # Analyze response distribution
        predictions = [row["model_choice"] for row in rows_out if row["valid_format"]]
        
        print(f"\nsaved → {out_csv}   ({total} rows)")
        print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
        print(f"Valid format: {valid_format}/{total} = {format_accuracy:.2f}%")
        print(f"Random baseline: {100/12:.2f}%")
        
        if predictions:
            from collections import Counter
            pred_counts = Counter(predictions)
            print(f"\nValid predictions distribution:")
            for letter in 'ABCDEFGHIJKL':
                count = pred_counts.get(letter, 0)
                if count > 0:
                    print(f"  {letter}: {count} times ({count/len(predictions)*100:.1f}%)")
            
            # Check for bias - flag if any letter appears >30% of the time
            max_freq = max(pred_counts.values()) if pred_counts else 0
            if max_freq > len(predictions) * 0.3:
                most_common = pred_counts.most_common(1)[0]
                print(f"\n⚠️  WARNING: Possible bias detected - '{most_common[0]}' appears {most_common[1]/len(predictions)*100:.1f}% of the time")
        
        # Show some examples of what went wrong
        invalid_responses = [row for row in rows_out if not row["valid_format"]]
        if invalid_responses:
            print(f"\nSample invalid responses:")
            for i, row in enumerate(invalid_responses[:3]):
                print(f"  {i+1}. Raw: '{row['model_choice_raw']}' → Extracted: '{row['model_choice']}'")
                
    else:
        print(f"saved → {out_csv}   (0 rows) - No successful predictions!")

# ---------- CLI --------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--history_type", choices=["yoked", "none"], default="yoked")
    p.add_argument("--data_path", default="trials_with_history.csv")
    p.add_argument("--image_path", default="compiled_grid.png")
    main(p.parse_args())