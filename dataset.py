import torch
from torch.utils.data import Dataset
from typing import Union
import util
import os
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, audio_file_path: Union[str, list, tuple], sampling_rate, frame_size, shift_size, window_type='uniform', dtype='float32', extension='wav'):
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.shift_size = shift_size
        self.window = util.window(window_type, self.frame_size, dtype)

        print("DATASET INITIATING!")
        # single audio dataset
        if isinstance(audio_file_path, str):
            all_audio_path_list = [util.read_path_list(audio_file_path, extension=extension)]

        # multiple audio dataset
        else:
            if os.path.isdir(audio_file_path[0]):
                if not util.compare_path_list(audio_file_path, extension=extension):
                    util.raise_error('Audio file lists are not same')
                all_audio_path_list = util.read_path_list(audio_file_path, extension=extension)
            else:
                all_audio_path_list = [[path] for path in audio_file_path]

        # make audio data list
        self.all_audio_data_list = []
        for path_list in all_audio_path_list:
            x_list = []
            for path in path_list:
                x = util.read_audio_file(path, sampling_rate=self.sampling_rate, different_sampling_rate_detect=True)
                x_list.append(x)
            x = np.concatenate(x_list)

            self.front_padding_size = self.frame_size
            self.rear_padding_size = (self.shift_size - (len(x) + self.frame_size + self.shift_size) % self.shift_size) % self.shift_size + self.frame_size
            x = np.pad(x, (self.front_padding_size, self.rear_padding_size), 'constant', constant_values=0.0).astype(dtype)
            self.number_of_frame = (len(x) - self.frame_size + self.shift_size) // self.shift_size
            self.total_length = len(x)
            self.all_audio_data_list.append(x)

        # check different data length
        for i in range(len(self.all_audio_data_list) - 1):
            if len(self.all_audio_data_list[i]) != len(self.all_audio_data_list[i + 1]):
                util.raise_error('Audio file length are not same')

    def __len__(self):
        return self.number_of_frame

    def __getitem__(self, index):
        return_list = [torch.from_numpy(data[index*self.shift_size:index*self.shift_size+self.frame_size]*self.window) for data in self.all_audio_data_list]
        return_list.insert(0, index)
        return return_list