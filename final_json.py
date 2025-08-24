import os
import json
import re
import ast
from tqdm import tqdm
import argparse


def load_predictions_from_folder(pred_folder: str):
    predictions = []
    for item in tqdm(os.listdir(pred_folder), desc="Reading predictions"):
        pred_path = os.path.join(pred_folder, item)
        if not os.path.isfile(pred_path):
            continue
        with open(pred_path, 'r') as f:
            raw_text = f.read().strip()
            try:
                pred_dict = ast.literal_eval(raw_text)
            except (SyntaxError, ValueError) as e:
                print(f"Skipping {item}, could not parse: {e}")
                continue
            pred_text = pred_dict.get('predict', "")
        predictions.append({"id": item.split('.tx')[0] + '.tiff', "report": pred_text})
    return predictions


def format_reports_like_ground_truth(data):
    modified_count = 0
    for record in data:
        if 'report' in record and isinstance(record['report'], str):
            original_text = record['report']
            text = original_text
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'\s*\+\s*', '+', text)
            text = re.sub(r'\s+([,;:)])', r'\1', text)
            text = re.sub(r'([(])\s+', r'\1', text)
            text = re.sub(r'([,])(?=[^\s])', r'\1 ', text)
            text = re.sub(r'([:])(?=[^\s])', r'\1 ', text)
            text = re.sub(r'(?<!\s)([(])', r' \1', text)
            text = text.replace(';', ';\n ')
            text = text.replace(' . ', '. ')
            if original_text != text:
                record['report'] = text
                modified_count += 1
    return data, modified_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert prediction text files in a folder to formatted JSON.")
    parser.add_argument('--pred_folder', type=str, required=True, help='Path to the folder containing prediction .txt files')
    args = parser.parse_args()

    pred_folder = args.pred_folder +'/reports'
    output_json_path = os.path.join(pred_folder, 'predictions.json')
    predictions = load_predictions_from_folder(pred_folder)
    predictions, modified_count = format_reports_like_ground_truth(predictions)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f)
    print(f"Processed {len(predictions)} records.")
    print(f"Re-formatted {modified_count} records.")
    print(f"Final JSON saved to '{output_json_path}'.")
