import pandas as pd
from sklearn.model_selection import train_test_split

from Tokenization import scripts_subset, labels_subset


# Apply label preprocessing
# Assuming 'tokens' and 'processed_labels' are your features and labels
train_tokens, test_tokens, train_labels, test_labels = train_test_split(scripts_subset, labels_subset, test_size=0.2, random_state=42)


