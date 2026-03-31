import argparse
from Training.full_training import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    print('Training...')
    train_full_model(True)
