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

    # dataloader
    frame_size = past_size + present_size + future_size
    test_set = dataset.AudioDataset([test_orig_path, test_noisy_path], sampling_rate, frame_size, shift_size, input_window)
    test_sampler = DistributedSampler(test_set, shuffle=False) if num_gpus > 1 else None
    test_loader = DataLoader(test_set, num_workers=num_gpus*4, shuffle=False, sampler=test_sampler, batch_size=batch_per_gpu, pin_memory=True, drop_last=True)

    # run
    wavenet.train()
    for epoch in range(saved_epoch + 1, saved_epoch + epochs + 1):
        if rank == 0:
            start = time.time()
            now_time = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
            num_of_iteration = math.ceil(train_set.number_of_frame/batch_size)
            train_mae_loss = 0

        if num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                print(f"\r({now_time})", end=" ")
                print(util.change_font_color('bright cyan', 'Train:'), end=" ")
                print(util.change_font_color('yellow', 'epoch'), util.change_font_color('bright yellow', f"{epoch}/{saved_epoch + epochs},"), end=" ")
                print(util.change_font_color('yellow', 'iter'), util.change_font_color('bright yellow', f"{i + 1}/{num_of_iteration}"), end=" ")

            index, orig, noisy = batch
            orig = orig.to(device, non_blocking=True)
            noisy = noisy.to(device, non_blocking=True)
            orig = orig.unsqueeze(1)
            noisy = noisy.unsqueeze(1)
            noise = noisy-orig

            optimizer.zero_grad()

            orig_pred = wavenet(noisy)
            noise_pred = noisy-orig_pred

            orig = orig[:, :, past_size:past_size+present_size]
            noise = noise[:, :, past_size:past_size+present_size]
            orig_pred = orig_pred[:, :, past_size:past_size+present_size]
            noise_pred = noise_pred[:, :, past_size:past_size+present_size]

            loss = (l1_loss(orig, orig_pred) + l1_loss(noise, noise_pred)) / 2.0 / batch_size

            loss.backward()
            optimizer.step()
            if num_gpus > 1:
                dist.all_reduce(loss)
            if rank == 0:
                train_mae_loss += loss.item()

        if rank == 0:
            # neptune log
            train_mae_loss /= num_of_iteration
            neptune.log('mae_loss', train_mae_loss, epoch)

            # checkpoint save
            if epoch % save_checkpoint_period == 0 or epoch == 1:
                save_checkpoint_path = f"./checkpoint/{save_checkpoint_name}_{epoch}.pth"
                os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
                checkpoint = {'wavenet': (wavenet.module if num_gpus > 1 else wavenet).state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, save_checkpoint_path)

            end_time = util.second_to_dhms_string(time.time() - start)
            print(util.change_font_color('bright black', '|'), end=" ")
            print(util.change_font_color('bright red', 'Loss:'), util.change_font_color('bright yellow', f"{train_mae_loss:.4E}"), end=" ")
            print(f"({end_time})")

        scheduler.step()

    neptune.stop()


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
        mp.spawn(train, nprocs=num_gpus, args=([num_gpus, batch_per_gpu],))
    else:
        train(0)
