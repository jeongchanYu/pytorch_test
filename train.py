import neptune.new as neptune
from api_key import *
from config import *
import util
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import model

# neptune preset
custom_run_id_len = len(f'{os.getlogin()}_{model_name}')
if custom_run_id_len>=30:
    util.raise_error(f'model_name is too long. You must delete {custom_run_id_len-29} characters.')

neptune_run = neptune.init(
    project="csp-lab/AEC",
    api_token=api_key['yjc'],
    source_files=['train.py', 'model.py', 'config.py', 'loss_function.py'],
    custom_run_id=f'{os.getlogin()}_{model_name}',
    name=model_name,
    tags=[os.getlogin(), model_name],
    flush_period=10,
    capture_stdout=False,
    capture_stderr=False
)

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

    if rank == 0:
        # LOAD MODEL
        pass

    if num_gpus > 1:
        generator = DistributedDataParallel(wavenet, device_ids=[rank]).to(device)


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

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
            optim_g.step()

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

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

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

