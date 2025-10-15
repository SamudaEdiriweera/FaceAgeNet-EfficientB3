"""
Entry point: runs training end-to-end.
You can later add CLI args (e.g., --epochs) and pass them into train.train_model().
"""
from src.train import train_model

if __name__ == "__main__":
    model, history, export_dir = train_model()
    print("Training complete, Exported SavedModel:", export_dir)