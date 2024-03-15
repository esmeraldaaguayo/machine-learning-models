import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10