import pandas as pd
from transformers import BertTokenizer
from main import scripts, labels  # Ensure that 'scripts' and 'labels' are correctly loaded in 'main.py'

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Select the first 61 samples
scripts_subset = scripts[:60]
labels_subset = labels[:60]

# Tokenize the subset of scripts
tokens = [tokenizer(script, padding='max_length', truncation=True, return_tensors='pt') for script in scripts_subset]

# Debugging: Print the first few tokenized scripts
print("Tokenized Scripts:")
print(tokens[:5])  # Print first 5 tokenized scripts to check

# Preprocess labels
def preprocess_label(label):
    if pd.isna(label):  # Handle missing labels
        return ['NoConnection']
    label = str(label)  # Ensure the label is a string
    connections = label.split('^')
    return connections

# Apply label preprocessing
processed_labels = [preprocess_label(label) for label in labels_subset]

# Debugging: Print the first few processed labels
print("Processed Labels:")
print(processed_labels[:5])  # Print first 5 processed labels to check

# Convert tokenized data to a format that can be saved in a DataFrame
# Convert tokens to a simple list of dictionaries for CSV storage
tokens_as_dict = [{'input_ids': token['input_ids'].tolist(),
                   'attention_mask': token['attention_mask'].tolist()}
                  for token in tokens]

# Create a DataFrame with the tokenized scripts and processed labels
df = pd.DataFrame({
    'scripts': scripts_subset,
    'tokens': tokens_as_dict,
    'labels': labels_subset,
    'processed_labels': processed_labels
})

# Save to a CSV file for later inspection
df.to_csv('tokenized.csv', index=False)
print("Tokenized subset saved to 'tokenized.csv'")