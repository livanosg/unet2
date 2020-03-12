from os import environ
import tensorflow as tf
from model_fn import ynet_model_fn
from input_fns import input_fn
from config import paths

model_path = paths['save'] + '/model'
eval_path = model_path + '/eval'

environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# TODO Implement on multi-nodes SLURM
strategy = tf.distribute.MirroredStrategy()
# If op cannot be executed on GPU ==> assign to CPU.
session_config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True)  # Avoid error message if there is no gpu available.
session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.
# Setting up working environment
warm_start = None
params = {'archit': 'ynet',
          'branch': 1,
          'mode': args.mode,
          'batch_norm': True,
          'dropout': 0.5,
          'classes': 2,
          'lr': 0.0001,
          'model_path': model_path,
          'eval_path': eval_path,
          'decay_rate': 0.1,
          'modality': 'CT',
          'augm_set': 'all',
          'augm_prob': 0.5,
          'batch_size': 1,
          'epochs': 200,
          'shuffle': False}
configuration = tf.estimator.RunConfig(model_dir=model_path,
                                       train_distribute=strategy,
                                       # eval_distribute=strategy, ==> breaks distributed training
                                       session_config=session_config)

model = tf.estimator.Estimator(model_fn=ynet_model_fn,
                               params=params,
                               config=configuration,
                               warm_start_from=warm_start)

if args.mode == 'make-labels':  # Prediction mode used for test data of CHAOS challenge
    params['shuffle'] = False
    predicted = model.predict(input_fn=lambda: input_fn(mode=tf.estimator.ModeKeys.PREDICT, params=params),
                              predict_keys=['predicted', 'path'], yield_single_examples=True)
    pairs = data_generator('eval', params=params, only_paths=False)
    for idx, output in enumerate(zip(predicted, pairs)):
        input = output[1][0]
        input = (input - np.min(input)) / (np.max(input) - np.min(input))
        results = output[0]['predicted'].astype(np.uint8)
        label = output[1][1]
        label[label == 255] = 1
        label_2 = label + results * 2
        label_2[label_2 == 3] = 0
        label_2 = label_2 / 2
        stack = np.hstack((input, results, label, label_2))
        cv2.imshow('Test', stack)
        cv2.waitKeyEx()
