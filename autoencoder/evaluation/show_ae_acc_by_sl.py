import os.path
import matplotlib.pyplot as plt
import sys
import pickle
from scipy.ndimage import uniform_filter1d
import matplotlib as mpl
import paths

sys.path.append("..")
from cadlib.macro import *

LOG_SCALE = False
ERROR_PLOTS = False

metric_plot_map = {
    "command_acc": 0,
    "param_acc": 1,
    "cd": 2,
    "ir": 4,
    "iou": 3
}

class AccHolder:
    def __init__(self, header, smooth=True, recon_name="reconstructions"):
        acc_path = paths.HOME_PATH + header + recon_name + "_acc_stat_by_seq_len.pkl"
        cd_path = paths.HOME_PATH + header + recon_name + "_pc_stat_by_seq_len.pkl"
        iou_path = paths.HOME_PATH + header + "eval_recon_iou_by_seq_len.pkl"

        self.metrics_sl = {}
        self.metrics_avg = {}

        # dic_save_path = results_folder

        with open(acc_path, 'rb') as f:
            output = pickle.load(f)

        avg_command_acc = output['avg_cmd_acc']
        avg_param_acc = output['avg_param_acc']

        self.command_accs = []
        self.param_accs = []
        self.seq_lens = sorted(list(avg_command_acc.keys()))
        for key in self.seq_lens:#range(3, 60):
            if LOG_SCALE or ERROR_PLOTS:
                self.command_accs.append(100 - np.mean(avg_command_acc[key]) * 100)
                self.param_accs.append(100 - np.mean(avg_param_acc[key]) * 100)
            else:
                self.command_accs.append(np.mean(avg_command_acc[key]) * 100)
                self.param_accs.append(np.mean(avg_param_acc[key]) * 100)

        self.command_accs = np.array(self.command_accs)
        self.param_accs = np.array(self.param_accs)

        self.avg_cmd = np.mean(self.command_accs)
        self.avg_param = np.mean(self.param_accs)

        self.metrics_sl["command_acc"] = self.command_accs
        self.metrics_sl["param_acc"] = self.param_accs
        self.metrics_avg["command_acc"] = self.avg_cmd
        self.metrics_avg["param_acc"] = self.avg_param

        if os.path.isfile(cd_path):
            with open(cd_path, 'rb') as f:
                output = pickle.load(f)

            avg_cd = output['cd']
            avg_ir = output['ir']

            self.cds = []
            self.irs = []
            for key in range(3, 60):
                self.cds.append(np.median(avg_cd[key]) * 1000)
                self.irs.append(np.mean(avg_ir[key]) * 100)

            self.cds = np.array(self.cds)
            self.irs = np.array(self.irs)

            self.avg_cd = np.mean(self.cds)
            self.avg_ir = np.mean(self.irs)

            self.metrics_sl["cd"] = self.cds
            self.metrics_sl["ir"] = self.irs
            self.metrics_avg["cd"] = self.avg_cd
            self.metrics_avg["ir"] = self.avg_ir

        if os.path.isfile(iou_path):
            with open(iou_path, 'rb') as f:
                output = pickle.load(f)

            avg_iou = output["avg_iou"]

            self.ious = []
            for key in range(3, 60):
                if key not in avg_iou.keys():
                    self.ious.append(0)
                    continue
                self.ious.append(np.mean(avg_iou[key]) * 100)

            self.ious = np.array(self.ious)
            self.avg_iou = np.mean(self.ious)

            self.metrics_sl["iou"] = self.ious
            self.metrics_avg["iou"] = self.avg_iou

        if smooth:
            filter = lambda x: uniform_filter1d(x, 5)

            for metric_key, value in self.metrics_sl.items():
                self.metrics_sl[metric_key] = filter(value)

if __name__=="__main__":
    accuracy_holders = {
        "*GenCAD-3D + SynthBalFT": AccHolder("results/Autoencoder/GenCAD3D_SynthBal_FT/autoencoder/"),
        "*GenCAD-3D + SynthBal-1MFT": AccHolder("../../results/GenCAD3D_SynthBal_1M_FT/autoencoder/"),
    }

    plot_args = {
        "*GenCAD-3D + SynthBalFT": {"c": "black"},
        "*GenCAD-3D + SynthBal-1MFT":  {"c": "mediumpurple"},
    }



    def mean_ignore_none(array):
        filter = array[np.invert(np.isnan(array))]
        return np.mean(filter)

    for key in accuracy_holders.keys():
        # accuracy_holders[key] = AccHolder(accuracy_holders[key])
        print(key, "#"*10)
        for metric_key, value in accuracy_holders[key].metrics_avg.items():
            print(metric_key, np.round(value, 2))

    # now plot
    fig, axs = plt.subplots(5, layout="constrained", figsize=(7, 10))

    label_fontsize = 22
    title_fontsize = 24
    legend_fontsize = 13

    font_family = "serif"

    def set_ax_options(ax):
        ax.grid(which='major')
        # ax.grid(True, which='minor', ls="--", alpha=0.5)
        ax.minorticks_on()
        ax.set_xlim(left=3, right=59)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        if LOG_SCALE:
            ax.set_yscale('log')

    ax = axs[0]
    if LOG_SCALE or ERROR_PLOTS:
        ax.set_title(r"Command Error ($\downarrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
        ax.set_ylabel("Error", fontsize=label_fontsize, family=font_family)
    else:
        ax.set_title(r"Command Accuracy ($\uparrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
        ax.set_ylabel("Accuracy", fontsize=label_fontsize, family=font_family)
    set_ax_options(ax)

    ax = axs[1]
    if LOG_SCALE or ERROR_PLOTS:
        ax.set_title(r"Parameter Error ($\downarrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
        ax.set_ylabel("Error", fontsize=label_fontsize, family=font_family)
    else:
        ax.set_title(r"Parameter Accuracy ($\uparrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
        ax.set_ylabel("Accuracy", fontsize=label_fontsize, family=font_family)    # if LOG_SCALE:

    set_ax_options(ax)

    ax = axs[2]
    ax.set_title(r"Chamfer Distances ($\downarrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
    ax.set_ylabel("CD", fontsize=label_fontsize, family=font_family)
    set_ax_options(ax)

    ax = axs[3]
    ax.set_title(r"IoU ($\uparrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
    ax.set_ylabel("IoU", fontsize=label_fontsize, family=font_family)
    set_ax_options(ax)

    ax = axs[4]
    ax.set_title(r"Invalid Ratio ($\downarrow$)", fontweight='bold', fontsize=title_fontsize, family=font_family)
    ax.set_ylabel("IR", fontsize=label_fontsize, family=font_family)
    set_ax_options(ax)



    ax.set_xlabel("Sequence Length", fontsize=label_fontsize, family=font_family)


    # x_val = list(range(3, 60))
    for key in accuracy_holders.keys():
        plotting_kwargs = {}
        if key in plot_args:
            plotting_kwargs = plot_args[key]
        x_val = accuracy_holders[key].seq_lens
        for metric_key, value in accuracy_holders[key].metrics_sl.items():
            ax = axs[metric_plot_map[metric_key]]
            ax.plot(x_val, value, label=key, **plotting_kwargs)

    mpl.rc('font', family=font_family)

    axs[1].legend(fontsize=legend_fontsize)
    plt.show()
