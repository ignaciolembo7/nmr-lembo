from data_analyzer import data_analyzer
from matplotlib import pyplot as plt
import numpy as np
import functions_aux
import seaborn as sns
import nogse
import os

class nogsevsx_processing_with_A0(data_analyzer):
    def analyze(self):

        mask = np.loadtxt(f"scripts/rois/mask_{self.mask_noise}.txt")  
        tnogse, g, n, x, TE, f_noise, err_f_noise, f_A0_noise, err_f_A0_noise = functions_aux.generate_NOGSE_vs_x_roi_with_A0(self.image_paths, self.method_paths, mask, self.slic)

        fig, ax = plt.subplots(figsize=(8,6))

        palette = sns.color_palette("tab10", len(self.rois_list))

        idx = 1 
        for roi, color in zip(self.rois_list, palette): 

            print("\nROI: ", roi, "\n")

            mask = np.loadtxt(f"scripts/rois/mask_{idx}.txt")

            fig1, ax1 = plt.subplots(figsize=(8,6))
            
            tnogse, g, n, x, TE, f, err_f, f_A0, err_f_A0 =  functions_aux.generate_NOGSE_vs_x_roi_with_A0_riciannoise(self.image_paths, self.method_paths, mask, self.slic, f_noise, f_A0_noise)

            print(f"NOGSE parameters for the {len(tnogse)} experiments:\n")
            print("T_nogse:\n",tnogse)
            print("g:\n",g)
            print("x:\n",x)
            print("N:\n",n)
            print("TE:\n",TE)

            directory = f"../{self.file_name}/slice={self.slic}/tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}"
            os.makedirs(directory, exist_ok=True)

            nogse.plot_nogse_vs_x_data(ax, roi, x, f, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f), tnogse[0], g[0], n[0], self.slic, color) 
            nogse.plot_nogse_vs_x_data(ax1, roi, x, f, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f), tnogse[0], g[0], n[0], self.slic, color)

            nogse.plot_A0_vs_x_data(ax1, roi, x, f_A0, ( np.std(f_A0_noise) / np.mean(f_A0_noise) ) * np.array(f_A0), tnogse[0], g[0], n[0], self.slic, color)

            table = np.vstack((x, f, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f)))
            np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}.txt", table.T, delimiter=' ', newline='\n')

            table = np.vstack((x, f_A0, ( np.std(f_A0_noise) / np.mean(f_A0_noise) ) * np.array(f_A0)))
            np.savetxt(f"{directory}/{roi}_data_A0_vs_x_tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}.txt", table.T, delimiter=' ', newline='\n')

            fig1.tight_layout()
            fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}.pdf")
            fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}.png", dpi=600)
            plt.close(fig1)
                
            with open(f"../{self.file_name}/slice={self.slic}/{roi}_parameters_vs_tnogse_g={self.num_grad}.txt", "a") as a:
                print(tnogse[0], g[0], np.round(np.mean(f),2), np.round(np.std(f),2), np.round(np.mean(f_A0),2), np.round(np.std(f_A0),2), file=a)

            with open(f"../{self.file_name}/slice={self.slic}/{roi}_parameters_vs_g_tnogse={tnogse[0]}.txt", "a") as a:
                print(tnogse[0], g[0], np.round(np.mean(f),2), np.round(np.std(f),2), np.round(np.mean(f_A0),2), np.round(np.std(f_A0),2), file=a)

        ax.plot(x, f_noise, "o-", linewidth = 2, markersize=7, color='r', label=f"Fondo - ({np.round(np.mean(f_noise),2)} $\pm$ {np.round(np.std(f_noise),2)})")
        ax.legend(title_fontsize=15, fontsize=15, loc='best')
        ax1.plot(x, f_noise, "o-", linewidth = 2, markersize=7, color='r', label=f"Fondo - ({np.round(np.mean(f_noise),2)} $\pm$ {np.round(np.std(f_noise),2)})")

        fig.tight_layout()
        fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}.pdf")
        fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse[0]}_g={g[0]}_N={int(n[0])}.png", dpi=600)
        plt.close(fig)