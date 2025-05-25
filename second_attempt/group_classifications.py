# Simplifies the labels, putting them in 4 big categories

import pandas as pd

def group_galaxies(label):
    """Simple grouping based on label prefixes"""
    if label.startswith('SB'):
        return 'Barred Spiral'
    elif label.startswith('S'):
        first_char = label[1].lower() if len(label) > 1 else ''
        if first_char in {'a', 'b', 'c', 'd', 'r'}:
            return 'Spiral'
    elif label.startswith('E'):
        return 'Elliptical'
    return 'Other/Unclassified'

# Load data
df = pd.read_csv('dataset/processed/image_labels.csv')

# Apply simple grouping and remove original label
df = df.assign(galaxy_class=df['label'].apply(group_galaxies)).drop(columns=['label'])

# Show distribution
print("Simplified Class Distribution:")
print(df['galaxy_class'].value_counts())

# Save new dataset (will only contain image_path and galaxy_class)
df.to_csv('dataset/processed/grouped_classifications.csv', index=False)