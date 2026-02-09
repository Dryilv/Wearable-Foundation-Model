import json
import argparse
import os

def convert_index(input_file, output_file):
    print(f"Loading original index from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    print(f"Original entries: {len(data)}")
    
    new_data = []
    seen_paths = set()
    
    # Process entries
    for item in data:
        path = item['path']
        
        # Deduplication: 
        # Since we deprecated 'row', multiple rows pointing to the same file 
        # should now be collapsed into a single entry for that file.
        if path in seen_paths:
            continue
            
        seen_paths.add(path)
        
        # Construct new entry without 'row'
        new_entry = {
            'path': path,
            'label': item.get('label', 0)
        }
        
        # Preserve other useful metadata if exists
        if 'len' in item:
            new_entry['len'] = item['len']
        if 'split' in item:
            new_entry['split'] = item['split']
            
        new_data.append(new_entry)
        
    print(f"Processed entries (unique paths): {len(new_data)}")
    print(f"Removed duplicates: {len(data) - len(new_data)}")
    
    # Save to output file
    try:
        with open(output_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        print(f"Successfully saved new index to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old index format (with rows) to new format (file-based)")
    
    # Default input path from previous context
    default_input = "/home/bml/storage/mnt/v-044d0fb740b04ad3/org/WFM/model/SharedPhysioTFMAE/train_index.json"
    
    parser.add_argument('--input', type=str, default=default_input, 
                        help='Path to the original JSON index file')
    parser.add_argument('--output', type=str, default='train_index_new.json',
                        help='Path to the output JSON file')
    
    args = parser.parse_args()
    
    convert_index(args.input, args.output)
