import argparse
import logging

from pipeline import TrainingPipeline

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, default="bert-base-cased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    logger.info(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    pipeline = TrainingPipeline(batch_size=args.batch_size, lr=args.lr, n_epochs=args.n_epochs, weight_decay=args.weight_decay)
    