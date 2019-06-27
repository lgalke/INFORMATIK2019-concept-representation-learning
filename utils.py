import pandas as pd

def save_embedding(labels, embedding, outfile):
    """ Saves an embedding given by (unique) labels and vectors to outfile """
    print("Dumping embeddings to", outfile)
    df = pd.DataFrame(data=embedding, index=labels)
    df.to_csv(outfile, index=True,header=False)


def load_embedding(csvfile, as_dataframe=False):
    """ Loads an embedding from a csv file """
    df = pd.read_csv(csvfile, header=None, index_col=0)
    if as_dataframe:
        return df
    return df.index.values, df.values
