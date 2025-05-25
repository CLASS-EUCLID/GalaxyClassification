import csv
from collections import Counter

csv_file_path = 'dataset/processed/image_labels.csv'

label_counts = Counter()

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label = row['label'].strip()
        label_counts[label] += 1

for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"{label}: {count}")
