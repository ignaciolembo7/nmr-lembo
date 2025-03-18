from data_analyzer import data_analyzer
from matplotlib import pyplot as plt
import numpy as np
import functions_aux
import seaborn as sns
import nogse
import os

class contrastvsg_processing_with_A0(data_analyzer):
    def analyze(self):

        fig, ax = plt.subplots(figsize=(8,6)) 

        mask = np.loadtxt(f"scripts/rois/mask_{self.mask_noise}.txt")  

        tnogse, g_contrast, n, x, TE, f_hahn_noise, err_hahn_noise, f_cpmg_noise, err_cpmg_noise, f_A0_hahn_noise, err_A0_hahn_noise, f_A0_cpmg_noise, err_A0_cpmg_noise =  functions_aux.generate_contrast_vs_g_roi_with_A0(self.image_paths, self.method_paths, mask, self.slic)

        f_hahn_noise = np.array(f_hahn_noise)
        f_cpmg_noise = np.array(f_cpmg_noise)
        f_A0_hahn_noise = np.array(f_A0_hahn_noise)
        f_A0_cpmg_noise = np.array(f_A0_cpmg_noise)

        palette = sns.color_palette("tab10", len(self.rois_list))

        idx = 1
        for roi, color in zip(self.rois_list, palette): 

            print("\nROI: ", roi, "\n")

            mask = np.loadtxt(f"scripts/rois/mask_{idx}.txt")

            fig1, ax1 = plt.subplots(figsize=(8,6))

            tnogse, g, n, x, TE, f_hahn, err_hahn, f_cpmg, err_cpmg, f_A0_hahn, err_A0_hahn, f_A0_cpmg, err_A0_cpmg = functions_aux.generate_contrast_vs_g_roi_with_A0_riciannoise(self.image_paths, self.method_paths, mask, self.slic, f_cpmg_noise, f_hahn_noise, f_A0_cpmg_noise, f_A0_hahn_noise)

            print(f"NOGSE parameters for the {len(tnogse)} experiments:\n")
            print("tnogse:",tnogse)
            print("g:",g)
            print("x:",x)
            print("N:",n)
            print("TE:",TE)

            f1 = f_cpmg / f_A0_cpmg
            f2 = f_hahn / f_A0_hahn
            err_f1 = f1 * np.sqrt((err_cpmg / f_cpmg)**2 + (err_A0_cpmg / f_A0_cpmg)**2)
            err_f2 = f2 * np.sqrt((err_hahn / f_hahn)**2 + (err_A0_hahn / f_A0_hahn)**2)
            f = f1 - f2
            err_f = np.sqrt(err_f1**2 + err_f2**2)

            directory = f"../{self.file_name}/slice={self.slic}/tnogse={tnogse[0]}_N={int(n[0])}"
            os.makedirs(directory, exist_ok=True)

            nogse.plot_contrast_vs_g_data(ax, roi, g, f, err_f, tnogse[0], n[0], self.slic, color) 
            nogse.plot_contrast_vs_g_data(ax1, roi, g, f, err_f, tnogse[0], n[0], self.slic, color)

            table = np.vstack((g, f_cpmg, err_cpmg, f_hahn, err_hahn, f_A0_cpmg, err_A0_cpmg, f_A0_hahn, err_A0_hahn))
            np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={tnogse[0]}_N={int(n[0])}.txt", table.T, delimiter=' ', newline='\n')

            fig1.tight_layout()
            fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse[0]}_N={int(n[0])}.pdf")
            fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse[0]}_N={int(n[0])}.png", dpi=600)
            plt.close(fig1)

            f_noise =  f_cpmg_noise/f_A0_cpmg_noise - f_hahn_noise/f_A0_hahn_noise

            idx += 1

        ax.plot(g, f_noise, "o-", linewidth = 2, markersize=7, color='r', label=f"Fondo - ({np.round(np.mean(f_noise),2)} $\pm$ {np.round(np.std(f_noise),2)})")
        ax.legend(title_fontsize=15, fontsize=15, loc='best')
        ax1.plot(g, f_noise, "o-", linewidth = 2, markersize=7, color='r', label=f"Fondo - ({np.round(np.mean(f_noise),2)} $\pm$ {np.round(np.std(f_noise),2)})")

        fig.tight_layout()
        fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse[0]}_N={int(n[0])}.pdf")
        fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse[0]}_N={int(n[0])}.png", dpi=600)
        plt.close(fig)