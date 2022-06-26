#####################################################################################
epochs                          = 10000
batch_size                      = 128
learning_rate                   = 0.0001
learning_rate_decay             = 0.999
#####################################################################################
load_checkpoint_name            = ""
save_checkpoint_name            = "WAVENET"
save_checkpoint_period          = 10
model_name                      = "WAVENET"
seed                            = 1234
#####################################################################################
past_size                       = 1600
present_size                    = 2200
future_size                     = 1600
dilation                        = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
shift_size                      = 2200
sampling_rate                   = 16000
input_window                    = 'uniform'
output_window                   = 'uniform'
#####################################################################################
train_noisy_path                = "/home/ubuntu/DATASET/AEC_SYNTH"
train_orig_path                 = "/home/ubuntu/DATASET/AEC_SYNTH"
test_noisy_path                 = "/home/ubuntu/DATASET/AEC_SYNTH"
test_orig_path                  = "/home/ubuntu/DATASET/AEC_SYNTH"
#####################################################################################
