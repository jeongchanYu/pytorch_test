import os
import time
import datetime
import math
import time
import loss_function
import neptune_function
from api_key import *
from config import *
import model
import dataset
import util
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader


def test(rank, params):
    num_gpus, batch_per_gpu = params
    if num_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:54321', world_size=num_gpus, rank=rank)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)

    wavenet = model.Wavenet(dilation).to(device)
    l1_loss = loss_function.l1_loss()

    # load model
    if load_checkpoint_name != "":
        saved_epoch = int(load_checkpoint_name.split('_')[-1])
        load_checkpoint_path = os.path.join('./checkpoint', f"{load_checkpoint_name}.pth")
        checkpoint = torch.load(load_checkpoint_path, map_location=device)
        wavenet.load_state_dict(checkpoint['wavenet'])
    else:
        util.raise_error("Check load_checkpoint_name.")

    # multi gpu model upload
    if num_gpus > 1:
        wavenet = DistributedDataParallel(wavenet, device_ids=[rank])

    # read file list
    if not util.compare_path_list([test_orig_path, test_noisy_path], 'wav'):
        util.raise_error('Audio file lists are not same')
    test_orig_file_list = util.read_path_list(test_orig_path, 'wav')
    test_noisy_file_list = util.read_path_list(test_noisy_path, 'wav')
    num_of_files = len(test_orig_file_list)
    for file_index, (test_orig_file, test_noisy_file) in enumerate(zip(test_orig_file_list, test_noisy_file_list)):
        # dataloader
        frame_size = past_size + present_size + future_size
        test_set = dataset.AudioDataset([test_orig_file, test_noisy_file], sampling_rate, frame_size, shift_size, input_window)
        test_sampler = DistributedSampler(test_set, shuffle=False) if num_gpus > 1 else None
        test_loader = DataLoader(test_set, num_workers=num_gpus*4, shuffle=False, sampler=test_sampler, batch_size=batch_per_gpu, pin_memory=True, drop_last=False)

        # run
        wavenet.eval()
        if rank == 0:
            start = time.time()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            num_of_iteration = math.ceil(test_set.number_of_frame/batch_size)
            test_mae_loss = 0

        with torch.no_grad():
            output_orig = torch.zeros(test_set.total_length).to(device, non_blocking=True)
            output_noise = torch.zeros(test_set.total_length).to(device, non_blocking=True)
            window = torch.from_numpy(util.window(output_window, present_size)).to(device, non_blocking=True)
            for i, batch in enumerate(test_loader):
                if rank == 0:
                    print(f"\r({now_time})", end=" ")
                    print(util.change_font_color('bright cyan', 'Train:'), end=" ")
                    print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{file_index+1}/{num_of_files},"), end=" ")
                    print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{i + 1}/{num_of_iteration}"), end=" ")

                index, orig, noisy = batch
                orig = orig.to(device, non_blocking=True)
                noisy = noisy.to(device, non_blocking=True)
                orig = orig.unsqueeze(1)
                noisy = noisy.unsqueeze(1)
                noise = noisy-orig

                orig_pred = wavenet(noisy)
                noise_pred = noisy-orig_pred

                orig = orig[:, :, past_size:past_size+present_size]
                noise = noise[:, :, past_size:past_size+present_size]
                orig_pred = orig_pred[:, :, past_size:past_size+present_size]
                noise_pred = noise_pred[:, :, past_size:past_size+present_size]

                loss = (l1_loss(orig, orig_pred) + l1_loss(noise, noise_pred)) / 2.0 / batch_size

                for b in range(orig.shape[0]):
                    idx = int(index[b])
                    output_orig[shift_size*idx+past_size:shift_size*idx+past_size+present_size] = orig_pred[b, 0] * window
                    output_noise[shift_size*idx+past_size:shift_size*idx+past_size+present_size] = noise_pred[b, 0] * window

                if num_gpus > 1:
                    dist.all_reduce(loss)

                if rank == 0:
                    test_mae_loss += loss.item()

            if num_gpus > 1:
                dist.all_reduce(output_orig)
                dist.all_reduce(output_noise)

            if rank == 0:
                test_mae_loss /= num_of_iteration

                output_orig = output_orig.cpu().numpy()
                output_noise = output_noise.cpu().numpy()
                save_orig_file_name = util.remove_base_path(test_orig_file, test_orig_path)
                save_noise_file_name = util.remove_base_path(test_noisy_file, test_noisy_path)
                save_orig_file_path = os.path.join('./test_result', load_checkpoint_name, 'orig', save_orig_file_name)
                save_noise_file_path = os.path.join('./test_result', load_checkpoint_name, 'noise', save_noise_file_name)
                util.write_audio_file(save_orig_file_path, output_orig[test_set.front_padding_size:len(output_orig) - test_set.rear_padding_size], sampling_rate)
                util.write_audio_file(save_noise_file_path, output_noise[test_set.front_padding_size:len(output_noise) - test_set.rear_padding_size], sampling_rate)

                end_time = util.second_to_dhms_string(time.time() - start)
                print(util.change_font_color('bright black', '|'), end=" ")
                print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{test_mae_loss:.4E}"), end=" ")
                print(f"({end_time})")



if __name__ == '__main__':
    # train var preset
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        num_gpus = torch.cuda.device_count()
        batch_per_gpu = batch_size // num_gpus
        print('Batch size per GPU :', batch_per_gpu)
    else:
        num_gpus = 0

    # run
    if num_gpus > 1:
        mp.spawn(test, nprocs=num_gpus, args=([num_gpus, batch_per_gpu],))
    else:
        test(0)
