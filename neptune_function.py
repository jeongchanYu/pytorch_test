import neptune.new as neptune
import os
import util

class Neptune:
    def __init__(self, project_name, model_name, api_key, file_names):
        super(Neptune, self).__init__()

        # neptune preset
        custom_run_id_len = len(f'{os.getlogin()}_{model_name}')
        if custom_run_id_len >= 30:
            util.raise_error(f'model_name is too long. You must delete {custom_run_id_len-29} characters.')

        if isinstance(file_names, str):
            file_names = [file_names]

        self.neptune_run = neptune.init(
            project=project_name,
            api_token=api_key,
            source_files=file_names,
            custom_run_id=f'{os.getlogin()}_{model_name}',
            name=model_name,
            tags=[os.getlogin(), model_name],
            flush_period=10,
            capture_stdout=False,
            capture_stderr=False
        )

    def loss_init(self, losses, epoch, category='train'):
        if isinstance(losses, str):
            losses = [losses]
        for loss in losses:
            if self.neptune_run.exists(f'loss/{category}/{loss}'):
                last_data = self.neptune_run[f'loss/{category}/{loss}'].fetch_values(False)
                self.neptune_run[f'loss/{category}/{loss}'].pop()
                list_last_data = last_data['value'].values.tolist()
                list_last_data_step = last_data['step'].values.tolist()
                for l in range(len(list_last_data)):
                    if list_last_data_step[l] <= epoch:
                        self.neptune_run[f'loss/{category}/{loss}'].log(list_last_data[l], step=list_last_data_step[l])

    def log(self, loss_name, loss, epoch, category='train'):
        self.neptune_run[f'loss/{category}/{loss_name}'].log(loss, step=epoch)