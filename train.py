import os
import time
import datetime
import math
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

# neptune init
neptune = neptune_function.Neptune(project_name='csp-lab/AEC', model_name=model_name, api_key=api_key['yjc'],
                          file_names=['train.py', 'model.py', 'config.py', 'loss_function.py'])

# train var preset
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    num_gpus = torch.cuda.device_count()
    batch_per_gpu = batch_size // num_gpus
    print('Batch size per GPU :', batch_per_gpu)
else:
    num_gpus = 0


def train(rank):
    if num_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:FREEPORT', world_size=num_gpus, rank=rank)

    torch.cuda.manual_seed(seed)
    device = torch.device('cuda:{:d}'.format(rank))

    wavenet = model.Wavenet(dilation).to(device)
    optimizer = torch.optim.AdamW(wavenet.parameters(), learning_rate)

    # load model
    if rank == 0:
        if load_checkpoint_name != "":
            saved_epoch = int(load_checkpoint_name.split('_')[-1])
            load_checkpoint_path = os.path.join('./checkpoint', f"{load_checkpoint_name}.pth")
            checkpoint = torch.load(load_checkpoint_path, map_location=device)
            wavenet.load_state_dict(checkpoint['wavenet'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            saved_epoch = 0

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay, last_epoch=saved_epoch)

    # neptune loss init
    neptune.loss_init(['mae_loss'], saved_epoch, 'train')

    # multi gpu model upload
    if num_gpus > 1:
        wavenet = DistributedDataParallel(wavenet, device_ids=[rank]).to(device)

    # dataloader
    frame_size = past_size + present_size + future_size
    train_set = dataset.AudioDataset([train_orig_path, train_noisy_path], sampling_rate, frame_size, shift_size, input_window)
    train_sampler = DistributedSampler(train_set) if num_gpus > 1 else None
    train_loader = DataLoader(train_set, num_workers=num_gpus*4, shuffle=True,
                              sampler=train_sampler,
                              batch_size=batch_per_gpu,
                              pin_memory=True,
                              drop_last=True)

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

            orig, noisy = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)

            optimizer.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optimizer.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                             else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                             else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)


            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


# train
if num_gpus > 1:
    mp.spawn(train, nprocs=num_gpus)
else:
    train(0)

