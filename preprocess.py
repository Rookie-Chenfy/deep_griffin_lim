import argparse
import os
from multiprocessing import cpu_count
import dgl_hp as hp
from tqdm import tqdm
import numpy as np


from frontend.preprocessor import _process_utterance


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')

    timesteps = sum([int(m[4]) for m in metadata])
    sr = hp.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), timesteps, hours))


def run_preprocess(args, hparams):
    np.random.seed(hparams.seed)
    input_dirs = os.path.join(args.base_dir, args.data)
    out_dir = os.path.join(args.base_dir, args.output)
    spec_clean = os.path.join(out_dir, 'spec_clean')
    spec_noisy = os.path.join(out_dir, 'spec_noisy')
    mag_clean = os.path.join(out_dir, 'mag_clean')

    os.makedirs(mag_clean, exist_ok=True)
    os.makedirs(spec_noisy, exist_ok=True)
    os.makedirs(spec_clean, exist_ok=True)

    # executor = ProcessPoolExecutor(max_workers=cpu_count())
    # futures = []
    metadata = []
    data_ids = os.listdir(input_dirs)

    for data in tqdm(data_ids):
        wav_path = os.path.join(input_dirs, data)
        wav_name = data.strip().split('.')[0]

        metadata.append(_process_utterance(hparams,spec_clean, spec_noisy,
                                           mag_clean, wav_name, wav_path))

    write_metadata(metadata, out_dir)


def main():
    print('starte preprocessing data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/home/chenfeiyang/end2end/myDeepGL')
    parser.add_argument('--data', default='data/biaobei2/wavs')
    parser.add_argument('--output', default='data/training_data')
    parser.add_argument('--j', type=int, default=cpu_count())
    args = parser.parse_args()

    run_preprocess(args, hp)


if __name__ == '__main__':
    main()