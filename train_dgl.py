import os
from torch import nn
import argparse
import dgl_hp as hp
from model.path import Paths
from model.dataset import get_vocoder_datasets
from model.display import simple_table
from model.deepGL import Opt_DeGLI


if __name__ == "__main__" :
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Deep Griffin Lim Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--step', '-s', type=int, help='the model to train total steps')
    parser.set_defaults(lr=hp.learning_rate)
    parser.set_defaults(batch_size=hp.batch_size)
    parser.set_defaults(step=hp.total_step)
    args = parser.parse_args()

    batch_size = args.batch_size
    total_step = args.step
    lr = args.lr

    paths = Paths(hp.data_path, hp.voc_model_id)

    train_set, test_set = get_vocoder_datasets(paths, batch_size)
    print('-------------------> Finsh load data <-------------------')

    model_loss = nn.L1Loss(reduction='none')

    voc_model = Opt_DeGLI(hp, paths, lr, model_loss)
    voc_model.restore(paths.voc_latest_weights)
    print('-------------------> restore model <-------------------')

    simple_table([('Remaining', str((total_step - voc_model.get_step()) // 1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr)])

    voc_model.train(train_set, test_set, total_step)

    print('Training Complete.')
