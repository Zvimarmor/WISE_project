import re
import json
import argparse
import csv
import sys

# Handle missing numpy (common on login nodes)
try:
    import numpy as np
except ImportError:
    class MockNumpy:
        def float64(self, x): return float(x)
        def float32(self, x): return float(x)
        def int64(self, x): return int(x)
        def int32(self, x): return int(x)
        def array(self, x): return list(x)
        def mean(self, x): return sum(x)/len(x) if x else 0
        def equal(self, x, y): return [a==b for a,b in zip(x,y)]
        # Types for isinstance checks
        integer = int
        floating = float
        ndarray = list
    np = MockNumpy()

def parse_logs(log_file):
    data = []
    
    with open(log_file, 'r') as f:
        log_content = f.read()

    # Regex to split entries by "INFO - <N> editing:"
    entries = re.split(r'INFO - \d+ editing:', log_content)
    
    # Skip the first chunk (before first edit)
    for entry in entries[1:]:
        # Find the dictionary block starting with '{'
        start_idx = entry.find('{')
        if start_idx == -1:
            continue
            
        dict_str = entry[start_idx:].strip()
        
        # Heuristic to find the end of the dictionary
        # We look for the last '}'
        end_idx = dict_str.rfind('}')
        if end_idx != -1:
            dict_str = dict_str[:end_idx+1]
        
        try:
            # Handle numpy types in the string representation using our np object
            # We also provide 'array' which is often printed as 'array([...])'
            record = eval(dict_str, {"np": np, "array": np.array})
            data.append(record)
        except Exception as e:
            # print(f"Error parsing entry: {e}")
            pass

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str, help='Path to the .err or .out or .txt log file')
    parser.add_argument('--output', type=str, default='debug_wise_extracted_results', help='Base name for output files')
    args = parser.parse_args()
    
    data = parse_logs(args.log_file)
    print(f"Propagated {len(data)} records.")
    
    if not data:
        print("No data found. Check the log file format.")
        return

    # Convert to standard JSON
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(f"{args.output}.json", 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
        
    print(f"Saved JSON to {args.output}.json")
    
    # Extract key fields for CSV
    flat_data = []
    for d in data:
        row = {
            'case_id': d.get('case_id'),
            'prompt': d.get('requested_rewrite', {}).get('prompt'),
            'target_new': d.get('requested_rewrite', {}).get('target_new'),
            # Pre
            # Handle list vs scalar for reliability scores
            'pre_acc': d.get('pre', {}).get('rewrite_acc', [0])[0] if isinstance(d.get('pre', {}).get('rewrite_acc'), list) else d.get('pre', {}).get('rewrite_acc', 0),
            'pre_text': d.get('pre', {}).get('fluency', {}).get('generated_text', [""])[0],
            # Post
            'post_acc': d.get('post', {}).get('rewrite_acc', [0])[0] if isinstance(d.get('post', {}).get('rewrite_acc'), list) else d.get('post', {}).get('rewrite_acc', 0),
            'post_text': d.get('post', {}).get('fluency', {}).get('generated_text', [""])[0],
        }
        flat_data.append(row)
        
    # Write CSV using built-in csv module (no pandas needed)
    if flat_data:
        keys = flat_data[0].keys()
        with open(f"{args.output}.csv", 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(flat_data)
        print(f"Saved CSV to {args.output}.csv")

if __name__ == "__main__":
    main()
