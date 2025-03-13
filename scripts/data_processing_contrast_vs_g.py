#NMRSI - Ignacio Lembo Ferrari - 05/11/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

rois = ["Fibra_grande","Fibra_chica","Agua_libre"]
file_name = "results_fantomas_20230302"
folder = "contrast_vs_g_data"
data_directory = f"C:/Users/ignacio/data/data_fantomas_20230302"
slic = 0 # slice que quiero ver
exp = 1 

image_paths, method_paths = nogse.upload_contrast_vs_g_data(data_directory, slic)

fig, ax = plt.subplots(figsize=(8,6)) 

mask = np.loadtxt(f"rois/mask_4.txt")
tnogse, g, n, f_hahn_noise, error_hahn_noise, f_cpmg_noise, error_cpmg_noise, f_A0_hahn_noise, error_A0_hahn_noise, f_A0_cpmg_noise, error_A0_cpmg_noise =  nogse.generate_contrast_vs_g_roi(image_paths, method_paths, mask, slic)

f_hahn_noise = np.array(f_hahn_noise)
f_cpmg_noise = np.array(f_cpmg_noise)
f_A0_hahn_noise = np.array(f_A0_hahn_noise)
f_A0_cpmg_noise = np.array(f_A0_cpmg_noise)

print(f_hahn_noise, f_cpmg_noise, f_A0_hahn_noise, f_A0_cpmg_noise)

f_noise =  (f_cpmg_noise + f_hahn_noise)/2

palette = sns.color_palette("tab10", len(rois)+1)
idxs = [1,2,3]

for roi, color, idx in zip(rois, palette, idxs): 

    mask = np.loadtxt(f"rois/mask_{idx}.txt")

    fig1, ax1 = plt.subplots(figsize=(8,6))
    
    tnogse, g, n, f_hahn, f_cpmg, f_A0_hahn, f_A0_cpmg =  nogse.generate_contrast_vs_g_roi_riciannoise(image_paths, method_paths, mask, slic, f_cpmg_noise, f_hahn_noise, f_A0_cpmg_noise, f_A0_hahn_noise)

    f_hahn = np.array(f_hahn)
    f_cpmg = np.array(f_cpmg)
    f_A0_hahn = np.array(f_A0_hahn)
    f_A0_cpmg = np.array(f_A0_cpmg)

    f =  f_cpmg/f_A0_cpmg - f_hahn/f_A0_hahn

    directory = f"../results_{file_name}/{folder}/slice={slic}/tnogse={tnogse}_N={int(n)}_exp={exp}"
    os.makedirs(directory, exist_ok=True)

    nogse.plot_contrast_vs_g_data(ax, roi, g, f, np.abs( ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f) ), tnogse, n, slic, color) 
    nogse.plot_contrast_vs_g_data(ax1, roi, g, f, np.abs( ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f) ), tnogse, n, slic, color)

    table = np.vstack((g, f, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f), f_cpmg, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f_cpmg), f_hahn, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f_hahn)))
    np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

f_noise =  f_cpmg_noise/f_A0_cpmg_noise - f_hahn_noise/f_A0_hahn_noise

ax.plot(g, f_noise, "o-", linewidth = 2, markersize=7, color=palette[3], label=f"Fondo - ({np.round(np.mean(f_noise),2)} $\pm$ {np.round(np.std(f_noise),2)})")
ax.legend(title_fontsize=15, fontsize=15, loc='best')
ax1.plot(g, f_noise, "o-", linewidth = 2, markersize=7, color=palette[3], label=f"Fondo - ({np.round(np.mean(f_noise),2)} $\pm$ {np.round(np.std(f_noise),2)})")

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig)