import argparse
from tensorflow.estimator import ModeKeys


PARSER = argparse.ArgumentParser(description='Train a model according to given hyperparameters.')

# Mode
PARSER.add_argument('-M', '--mode', type=str, default=ModeKeys.TRAIN,
                    choices=[ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT, 'train-and-eval', 'test'],  # TODO EXPORT MODEL
                    help='Define the estimator mode')
# Model options
PARSER.add_argument('-load', '--load_model', type=str, default='', help=' If declared, the model saved will be loaded.')
PARSER.add_argument('-resume', action='store_true', default=False, help='Continue training from loaded model.')

PARSER.add_argument('-no_bn', action='store_false', default=True, help='Batch Normalization.')
PARSER.add_argument('-dr', '--dropout', type=float, default=0.5, help='Dropout rate.')
PARSER.add_argument('-cls', '--classes', type=int, default=2, choices=[2], help='Choose 2classes')

PARSER.add_argument('-lr', '--lr', type=float, default=0.0001, help='Learning Rate.')
PARSER.add_argument('-dc', '--decays_per_train', type=int, default=1, help='Times to decay learning rate.')
PARSER.add_argument('-dcr', '--decay_rate', type=float, default=0.1, help='Decay rate for learning rate.')

# Training dataset options
PARSER.add_argument('-modal', '--modality', type=str, default='MR', choices=['CT', 'MR', 'ALL'], help='Set type of training data.')
PARSER.add_argument('-aug', '--augm_set', type=str, default='all', choices=['geom', 'dist', 'all', 'none'], help='Define the augmentation type')
PARSER.add_argument('-augp', '--augm_prob', type=float, default=0.5, help='Probability for augmented image.')
PARSER.add_argument('-shfl', '--shuffle', type=bool, default=True, help='Shuffle data')
PARSER.add_argument('-batch', '--batch_size', type=int, default=2, help='Mini-batch size.')

# Training hyper-parameters
PARSER.add_argument('-e', '--epochs', type=int, default=100, help='Training epochs.')
PARSER.add_argument('-es', '--early_stop', type=int, default=20, help='Epochs without minimizing target.')
ARGS = PARSER.parse_args()

if __name__ == '__main__':
    from train import train_fn
    train_fn(ARGS)
