import json

def convert_json_format(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create the new format
    new_data = []
    
    for i, item in enumerate(data):

        if i % 1000 == 0:
            print(f"Processing item {i} of {len(data)}")

        conversations = item.get("conversations", [])
        
        # Check if we have at least two elements in conversations
        if len(conversations) >= 2:
            # Extract the prompt and completion
            prompt = conversations[0].get("value", "")
            completion = conversations[1].get("value", "")
            
            # Add to the new data structure
            new_data.append({
                "prompt": prompt,
                "completion": completion
            })
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Converted {len(new_data)} items.")
    return new_data

# Example usage
if __name__ == "__main__":
    input_file = "/ephemeral/datasets/dojo_datasets/dojo_sft.json"
    output_file = "/ephemeral/grpo-data/dojo_sft_js_grpo-format.json"
    convert_json_format(input_file, output_file)