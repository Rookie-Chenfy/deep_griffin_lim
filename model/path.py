import os


class Paths :
    def __init__(self, data_path, voc_id) :
        # Data Paths
        self.data = f'{data_path}/'
        self.input_data = f'{data_path}/train.txt'
        self.sp_cl = f'{self.data}spec_clean/'
        self.sp_no = f'{self.data}spec_noisy/'
        self.mag_cl = f'{self.data}mag_clean/'
        # Deep Griffin Lim/Vocoder Paths
        self.voc_checkpoints = f'checkpoints/{voc_id}/'
        self.voc_latest_weights = f'{self.voc_checkpoints}latest_weights.pyt'
        self.voc_output = f'outputs/{voc_id}/'
        self.voc_step = f'{self.voc_checkpoints}/step.npy'
        self.voc_log = f'{self.voc_checkpoints}log.txt'

        self.create_paths()

    def create_paths(self) :
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.sp_cl, exist_ok=True)
        os.makedirs(self.sp_no, exist_ok=True)
        os.makedirs(self.mag_cl, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)


