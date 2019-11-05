import os
import numpy as np
from frontend.audio_preprocess import load_wav
from frontend.audio_preprocess import  _stft, _db_to_amp


def _process_utterance(hparams, spec_clean, spec_noisy, mag_clean, wav_name, wav_path):
    try:
        # Load the audio as numpy array
        wav = load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    wav_len = len(wav)

    sp_cl = _stft(wav, hparams)
    mag_cl = np.abs(sp_cl)[..., np.newaxis]

    # 产生随机噪声
    signal_power = np.mean(np.abs(wav) ** 2)
    snr_db = -6 * np.random.rand()
    snr = _db_to_amp(snr_db)
    noise_power = signal_power / snr
    noisy = wav + np.sqrt(noise_power) * np.random.randn(len(wav))
    sp_no = _stft(noisy, hparams)


    spec_clean_filename = 'spec_clean_{}.npy'.format(wav_name)
    spec_noisy_filename = 'spec_noisy_{}.npy'.format(wav_name)
    mag_clean_filename = 'mag_clean_{}.npy'.format(wav_name)

    np.save(os.path.join(spec_clean, spec_clean_filename), sp_cl, allow_pickle=False)
    np.save(os.path.join(spec_noisy, spec_noisy_filename), sp_no, allow_pickle=False)
    np.save(os.path.join(mag_clean, mag_clean_filename), mag_cl, allow_pickle=False)

    # Return a tuple describing this training example
    return (spec_clean_filename, spec_noisy_filename, mag_clean_filename, snr_db, wav_len)



