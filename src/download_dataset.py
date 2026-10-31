
from datasets import load_dataset
import pandas as pd

# Download Dreaddit
dataset = load_dataset("andreagasparini/dreaddit")

# Check available splits
print(dataset)

# Combine available splits into CSV
df = pd.concat(
    [dataset[split].to_pandas().assign(split=split)
     for split in dataset.keys()],
    ignore_index=True
)

# Save CSV
df.to_csv("dreaddit.csv", index=False)

print("Dataset downloaded successfully!")
print(df.head())
print(df.shape)
