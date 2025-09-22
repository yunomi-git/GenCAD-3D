import matplotlib.pyplot as plt
from utils.util import DictList
import torch
import paths

class Logger:
    def __init__(self, save_dir, exp_name=None):
        self.logs = {} # list of dictlists
        self.epochs = {}
        self.save_dir = save_dir
        if exp_name is None:
            exp_name = ""
        self.exp_name = exp_name

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        paths.mkdir(save_dir)
    
    def add_to_log(self, log_name, metric_name, value, epoch):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if log_name not in self.logs.keys():
            self.logs[log_name] = DictList()
            self.epochs[log_name] = DictList()
        self.logs[log_name].add_to_key(metric_name, value)
        self.epochs[log_name].add_to_key(metric_name, epoch)

    # def add_dict_to_log(self, log_name, metric_dict, epoch):
    #     # may need to convert to numpy if tensor
    #     cleaned_dict = {}
    #     for key, value in metric_dict.items():
    #         if isinstance(value, torch.Tensor):
    #             cleaned_dict[key] = value.detach().cpu().numpy()
    #         else:
    #             cleaned_dict[key] = value
    #     if log_name not in self.logs.keys():
    #         self.logs[log_name] = DictList()
    #     self.logs[log_name].add_dict(cleaned_dict)
    #     self.epochs[log_name].add_to_key(metric_name, epoch)

    def plot_log(self, log_name):
        if log_name not in self.logs.keys():
            return
        log = self.logs[log_name]
        names = list(log.keys())
        values = list(log.values())
        epochs = list(self.epochs[log_name].values())
        plot_report(values, names, epochs=epochs, title=self.exp_name + "_" + log_name, save_name=log_name, save_dir=self.save_dir)        


def plot_report(arrays, names, title, save_dir, epochs, save_name):
    plt.figure(figsize=[16,12])

    for epoch, array in zip(epochs, arrays):
        plt.plot(epoch, array, marker='o')  # Plot the line with 'o' as the marker for points
        for ep, value in zip(epoch, array):
            if ep % 20 == 0:
                plt.text(ep, value, f'{value:.3f}', color = 'black', ha = 'center', va = 'bottom')

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(title, fontsize=22)
    plt.yscale('log')
    plt.legend([names[i] for i in range(len(names))], fontsize=14)
    plt.grid(True, which='both')
    plt.savefig(save_dir + f'{save_name}.png')
    # print("saved to", save_dir + f'{title}.png')
    plt.close()