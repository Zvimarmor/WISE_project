import json
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert WISE evaluation JSON to CSV.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("Error: JSON data is empty.")
        return

    # Extract headers from the first item
    headers = list(data[0].keys())
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            # Ensure all headers exist in the row (fill missing with None)
            writer.writerow({h: row.get(h, "") for h in headers})

    print(f"Successfully converted {args.input} to {args.output}")

if __name__ == "__main__":
    main()
