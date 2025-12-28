"""Main entry point - runs training and evaluation for all models."""

import pandas as pd

import config
from dataset import create_loaders
from train import train_model
from evaluate import run_evaluation


def load_data():
    df = pd.read_csv(config.CSV_FILENAME)
    sample_ids = config.SAMPLE_FLY_IDS + config.SAMPLE_NOFLY_IDS
    df = df[~df['id'].isin(sample_ids)]

    def build_filepath(row):
        if row['folder'] == 'fly':
            return config.FLY_FOLDER / row['filename']
        return config.NOFLY_FOLDER / row['filename']

    df['filepath'] = df.apply(build_filepath, axis=1)
    df = df[df['filepath'].apply(lambda p: p.exists())]

    n_fly, n_nofly = (df['label'] == 1).sum(), (df['label'] == 0).sum()
    print(f"Dataset: {len(df)} samples (Fly: {n_fly}, NoFly: {n_nofly}, Ratio: 1:{n_nofly/n_fly:.1f})")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_end = int(0.8 * len(df))
    val_end = train_end + int(0.1 * len(df))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:], n_nofly / n_fly


def main():
    print(f"{'='*50}\nBIRD FLIGHT CLASSIFIER\n{'='*50}")
    print(f"Device: {config.DEVICE} | Image: {config.IMAGE_SIZE}px | Epochs: {config.NUM_EPOCHS}")

    train_df, val_df, test_df, class_weight = load_data()
    train_loader, val_loader, test_loader = create_loaders(
        train_df, val_df, test_df, config.IMAGE_SIZE, config.BATCH_SIZE, config.NUM_WORKERS)

    histories = {}
    for name in config.MODELS:
        try:
            history, _, _ = train_model(name, train_loader, val_loader, class_weight)
            histories[name] = history
        except Exception as e:
            print(f"ERROR training {name}: {e}")

    run_evaluation(test_loader, histories)


if __name__ == "__main__":
    main()