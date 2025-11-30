import pandas as pd
import json
files = {
    'train': 'datasets/Roman_Urdu/csv/train_data_roman.csv',
    'val': 'datasets/Roman_Urdu/csv/validation_data_roman.csv'
}
def format_to_jsonl(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful medical AI assistant. You answer questions in Roman Urdu."
                        },
                        {
                            "role": "user",
                            "content": str(row['short_question']) # Ensure it's a string
                        },
                        {
                            "role": "assistant",
                            "content": str(row['short_answer'])
                        }
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Success: Converted {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error with {input_path}: {e}")

for split, filename in files.items():
    try:
        format_to_jsonl(filename, f'{split}_dataset.jsonl')
    except:
        pass