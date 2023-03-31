"""
Experiment is a message broker, passing messages to subscribers
"""

import os
import getpass

from .utils import *

class MessageBroker:
    """
    A MessageBroker is the main message coordinator
    """
    
    def __init__(self, subscribers=None):
        if subscribers is None:
            subscribers = []
        self.subscribers = subscribers
        
    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        
    def publish(self, channel, message): 
        if not isinstance(message, dict):
            raise TypeError("message type error. Publish expects message of type dict.")
            
        for subscriber in self.subscribers:
            subscriber.receive(message, channel)
            
class Experiment(MessageBroker):
    """
    An Experiment acts as a broker, managing the communication between computation and the record maintenance subscribers, and maintains experiment state, as the single authority on the current experiment directory and the current trial directory
    """
    def __init__(self, experiment_name, docengine_output_directory, subscribers=None):
        super(Experiment, self).__init__(subscribers)
        self.experiment_name = experiment_name
        self.trial_path = None
        self.trial_name = None
        self.docengine_output_directory = docengine_output_directory
        self.experiment_dir = os.path.join(docengine_output_directory, experiment_name)
        
    def _new_trial_name(self):
        return f"{getpass.getuser()}_{os.getpid()}_{utils.n_year_index()}"
    
    def expand_to_trial_dir(self, filename):
        if not self.trial_path:
            raise AttributeError("trial_path must be set before publishing Set trial_path with experiment.initialize_new_trial().")
        return os.path.join(self.trial_path, filename)
    
    def initialize_new_trial(self):
        self.trial_name = self._new_trial_name()
        self.trial_path = os.path.join(self._experiment_dir, self.trial_name)
        if not os.path.exists(dir_name):
            os.makedirs(dirname)
            
        for subscriber in self.subscribers:
            subscriber.set_trial_dir(self.trial_path)
            
    def add_subscriber(self, subscriber):
        subscriber.set_trial_dir(self.trial_path)
        super(Experiment, self).add_subscriber(subscriber)
        
    def publish(self, channel, message):
        if not self.trial_path:
            raise AttributeError("trial_path must be set before publishing. Set trial_path with experiment.initialize_new_trial().")
            
        super(Experiment, self).publish(channel, message)
        
    def log_metric(self, **metric_dict):
        self.publish(
            message=metric_dict, 
            channel="metric_events"
        )
        
    def log_epoch_metric(self, **metric_dict):
        metric_dict["metric_type"] = "epoch"
        self.publish(
            message=metric_dict, 
            channel="metric_events"
        )
        
    def log_batch_metric(self, **metric_dict):
        metric_dict["metric_type"] = "batch"
        self.publish(
            message=metric_dict, 
            channel="metric_events"
        )
        
    def log_hyperparams(self, hyperparams_dict):
        self.publish(
            message=hyperparams_dict, 
            channel="hyperparam_events"
        )
        
    def log_training_start(self):
        self.publish(
            message={"training_start": True}, 
            channel="training_events"
        )
        
    def log_training_end(self):
        self.publish(
            message={"training_end": True}, 
            channel="training_events"
        )
        
    def log_epoch_start(self):
        self.publish(
            message={"epoch_start": True}, 
            channel="training_events"
        )
        
    def log_epoch_end(self):
        self.publish(
            message={"epoch_end": True}, 
            channel="training_events"
        )
        
    def log_active_split(self, split):
        self.publish(
            message={"active_split": split}, 
            channel="training_events"
        )
        
    def log_model_output(self, **model_output_dict):
        self.publish(
            message=model_output_dict, 
            channel="model_output_events"
        )
        
def initialize_new_experiment_trial(experiment_name, subscribers=None, output_directory=None):
    output_directory = output_directory or settings.get_output_directory()
    default_subscribers = [LogSubscriber(to_console=False), DBSubscriber()]
    if subscribers is None:
        subscribers = default_subscribers
        
    experiment = Experiment(
        experiment_name=experiment_name, 
        docengine_output_directory=output_directory, 
        subscribers=subscribers
    )
    experiment.initialize_new_trial()
    return experiment
