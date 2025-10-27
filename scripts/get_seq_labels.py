import pandas as pd
from seq_fetch import *

def update_sequences_and_labels():
    df = pd.read_csv("./data/data.csv")
    sequences = []
    labels = []
    for _, row in df.iterrows():
        sequences.append(get_seq(row['insert_chrom'], row['insert_start'], row['insert_end']))
        if row['rna_dna_ratio'] < 1:
            labels.append(0)
        else:
            labels.append(1)

    df['sequence'] = sequences
    df['label'] = labels
    df.to_csv('./data/data_sequences.csv', index=False)
    return df

if __name__ == "__main__":
    update_sequences_and_labels()
    print("Sequences and labels updated and saved to './data/data_sequences.csv'.")