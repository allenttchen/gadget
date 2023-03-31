from .experiment import initialize_new_experiment_trial

def initialize(experiment_name, subscribers=None, output_directory=None):
    return initialize_new_experiment_trial(experiment_name, subscribers, output_directory)