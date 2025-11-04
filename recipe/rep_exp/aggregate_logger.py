import os
import json

class JsonEvalLogger:
    """
    A logger that logs to a json file.
    Args:
        resume_from_path: The path to the checkpoint to resume from, used to get the experiment name and checkpoint type.
        task: The task name, used to name the experiment.
    """
    def __init__(self, resume_from_path: str, task: str):
        self.root = 'eval'
        if resume_from_path is not None and resume_from_path != '':
            self.experiment_name = resume_from_path.split('/')[-2]
            self.checkpoint_type = resume_from_path.split('/')[-1]
        else:
            self.experiment_name = f'{task}_untrained'
            self.checkpoint_type = ''

    def flush(self):
        pass

    def log(self, data, step):
        # Create eval folder
        save_folder = os.path.join(self.root, self.experiment_name, self.checkpoint_type)
        os.makedirs(save_folder, exist_ok=True)

        # Save to json
        with open(os.path.join(save_folder, f"eval.json"), "w") as f:
            json.dump(data, f)