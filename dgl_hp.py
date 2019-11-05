
######## data preprocess hp#######
num_snr: int = 1
seed: int = 2019

######## wav hp ###################

num_mels: int = 80          # Number of mel-spectrogram channels and local conditioning dimensionality
num_freq: int = 1025         # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
n_fft: int = 2048           # Extra window size is filled with 0 paddings to match this parameter
hop_size: int = 275        # For 22050Hz, 275 ~= 12.5 ms 这里即为帧移
win_size: int = 1100        # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) 这里为帧长，2400/48000=0.05,即帧长为50ms
sample_rate: int = 22050   # 22050 Hz (corresponding to ljspeech dataset)
preemphasis: float = 0.97     # preemphasis coefficient


# Deep Griffin Lim
power: float = 1.2

data_path = 'data/training_data'
voc_model_id = 'yiwise'

######### train hp ################
voc_test_samples = 5
batch_size: int = 4
learning_rate: float = 5e-4
weight_decay: float = 1e-3
repeat_train = 1
thr_clip_grad = 4.0
total_step = 10000
voc_checkpoint_every = 1000

######### model hp ################
hop_length= 275
ch_hidden = 16
depth = 2
out_all_block = True