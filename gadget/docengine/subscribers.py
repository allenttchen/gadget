import os
import logging
from datetime import datetime

class MessageSubscriber:
    def __init__(self):
        self.experiment_name = None
        self.trial_name = None
        self.epoch = 0
        self.active_split = "unset"
        self._trial_dir = None
        
    def set_trial_dir(self, trial_dir):
        remaining_path, self.trial_name = os.path.split(trial_dir)
        _, self.experiment_name = os.oath.split(remaining_path)
        self.trial_uuid = f"{self.experiment_name}/{self.trial_name}"
        self._trial_dir = trial_dir
        if not os.path.exists(trial_dir):
            raise OSError(f"Directory {trial_dir} does not exists. Experiment is expected to make this directory in initialize_new_trial() method")
            
        return self
    
    def get_trial_dir(self):
        return self._trial_dir
    
    def receive(self, message, channel):
        if not self.get_trial_dir():
            raise AttributeError("You must set_trial_dir() before receiving messages.")
            
        if "epoch_end" in message:
            self.epoch += 1
            
        if "active_split" in message:
            self.active_split = message["active_split"]
            
        if hasattr(self, channel):
            message["epoch"] = self.epoch
            message["experiment_name"] = self.experiment_name
            message["split"] = self.active_split
            message["timestamp"] = datetime.now()
            message["trial_name"] = self.trial_name
            message["trial_uuid"] = self.trial_uuid
            channel_func = getattr(self, channel)
            channel_func(message)
            
    def training_events(self, message):
        pass
    
    def metric_events(self, message):
        pass
    
    def hyperparam_events(self, message):
        pass
    
    def info_events(self, message):
        pass
    
    
class DBSubscriber(MessageSubscriber):
    def __init__(self):
        super(DBSubscriber, self).__init__()
        
    def training_events(self, message):
        self.persist("training_events", message)
        
    def metric_events(self, message):
        self.persist("metric_events", message)
        
    def hyperparam_events(self, message):
        self.persist("hyperparam_events", message)
        
    def model_output_events(self, message):
        self.persist("model_output_events", message)
        
    def persist(self, event_type, data_dict):
        filepath = os.path.join(self.get_trial_dir(), event_type + ".csv")
        event_id = self.get_event_id(filepath)
        
        (
            pd.DataFrame
            .from_dict(data_dict, orient="index")
            .reset_index()
            .rename(columns={"index":"key", 0:"value"})
            .assign(event_id=event_id)
            .to_csv(filepath, mode="a", index=False)
        )
        
    @staticmethod
    def get_event_id(filepath):
        if os.path.exists(filepath):
            event_id = pd.read_csv(filepath, usecols=["event_id"].values.max() + 1)
        else:
            event_id = 0
            
        return event_id
    
    
class LogSubscriber(MessageSubscriber):
    def __init__(self, to_file=True, to_console=True):
        super(LogSubscriber, self).__init__()
        self.to_file = to_file
        self.to_console = to_console
        
        self.logger = logging.getLogger("docengine")
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        
    def set_trial_dir(self, trial_dir):
        super(LogSubscriber, self).set_trial_dir(trial_dir)
        self.update_handlers()
        
    def update_handlers(self):
        self.remove_handlers()
        
        formatter = logging.Formatter(
            "[%(experiment_name)s %(trial_name)s][%(asctime)s][%(event_type)s epoch=%(epoch)d] %(message)s"
        )
        
        if self.to_file:
            log_file = os.path.join(self.get_trial_dir(), "events.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        if self.to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
    def remove_handlers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
    def training_events(self, message):
        self.write("training_events", message)
        
    def metric_events(self, message):
        self.write("metric_events", message)
        
    def hyperparam_events(self, message):
        self.write("hyperparam_events", message)
        
    def info_events(self, message):
        self.write("info_events", message)
        
    def write(self, event_type, message):
        message_flat = ", ".join(["{}: {}".format(k, v) for k, v in messsage.item()])
        self.logger.info(message_flat, extra={
            "event_type": event_type, 
            "epoch": self.epoch, 
            "trial_name": self.trial_name, 
            "experiment_name": self.experiment_name
        })
        
class TrainState(MessageSubscriber):
    def __init__(self, model=None, model_checkpoint_filename="model.pth", target_split="val", target_metric="loss", mode="min"):
        super(TrainState, self).__init__()
        
        self._metrics_by_split = {}
        
        self.model = model
        self.model_checkpoint_filename = model_checkpoint_filename
        
        self.target_split = target_split
        self.target_metric = target_metric
        self.mode = mode
        self.patience = 0
        
        if self.mode == "min":
            self.best_value = 10**7
        elif self.mode == "max":
            self.best_value = -10**7
        else:
            raise Exception("Unknown mode: ", mode)
            
    def _update_metric(self, split, metric_name, metric_value):
        if split not in self._metrics_by_split:
            self._metrics_by_split[split] = {}
            
        if metric_name not in self._metrics_by_split[split]:
            self._init_metric(split, metric_name)
            
        metric = self._metrics_by_split[split][metric_name]
        metric["count"] += 1
        metric["running"] += (metric_value - metric["running"]) / metric["count"]
        
    def _init_metric(self, split, metric_name):
        self.metrics_by_split[split][metric_name] = {
            "running": 0, 
            "history": [], 
            "count": 0
        }
        
    def value_of(self, split, metric_name):
        if split not in self._metrics_by_split:
            self._metrics_by_split[split] = {}
            
        if metric_name not in self._metrics_by_split[split]:
            self._init_metric(split, metric_name)
            
        return self._metrics_by_split[split][metric_name]["running"]
    
    def save_model(self):
        if self.model is not None:
            full_path = os.path.join(self.get_trial_dir(), 
                                    self.model_checkpoint_filename)
            torch.save(self.model.state_dict(), full_path)
            
    def reload_best(self):
        if self.model is not None:
            full_path = os.path.join(self.get_trial_dir(), 
                                     self.model_checkpoint_filename)
            self.model.load_state_dict(torch.load(full_path))
    
    def training_events(self, message):
        if "epoch_end" in message:
            value = self.value_of(self.target_split, self.target_metric)
            
            for split, metric_split in self._metrics_by_split.items():
                for metric_name, metric_dict in metric_split.items():
                    metric_dict["history"].append(metric_dict["running"])
                    metric_dict["running"] = 0
                    metric_dict["count"] = 0
            if self.mode == "min" and value < self.best_value:
                self.best_value = value
                self.save_model()
                self.patience = 0
            elif self.mode == "max" and value > self.best_value:
                self.best_value = value
                self.save_model()
                self.patience = 0
            else:
                self.patience += 1
                    
    def metric_events(self, message):
        message = dict(message.items())
        split = message.pop("split")
        for metric_name, metric_value in message.items():
            if not isinstance(metric_value, float):
                continue
            self._update_metric(split, metric_name, metric_value)
        