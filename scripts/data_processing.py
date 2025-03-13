import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
import glob 

# Configuración de estilo
def set_style():
    sns.set_theme(context='paper')
    sns.set_style("whitegrid")
set_style()

# Configuración de widgets
layout = widgets.Layout(width='100%')
style = {'description_width': 'initial'}

# Definición de widgets
file_name = widgets.Text(value="results_fantomas_20230302", description="File Name:", layout=layout, style=style)
data_directory = widgets.Text(value="C:/Users/ignacio/data/data_fantomas_20230302", description="Data Dir:", layout=layout, style=style)
folder = widgets.Text(value="contrast_vs_g_data", description="Folder:", layout=layout, style=style)
rois_list = widgets.Text(value="Fibra_grande,Fibra_Chica, Agua", description="ROI:", layout=layout, style=style)
masks_list = widgets.Text(value="1", description="Mask:", layout=layout, style=style)
mask_noise = widgets.Text(value="4", description="Mask Noise:", layout=layout, style=style)
slic = widgets.IntText(value=0, description="Slice:", layout=layout, style=style)
folder_ranges = widgets.Text(value='34-45,47-58', description='Ingrese un conjunto de rangos de carpetas, por ejemplo, 106-108,110-115, ... :', layout=layout, style=style)

# Widgets de salida y botón
output = widgets.Output()
run_button = widgets.Button(description="Run Data Processing", layout=layout)

# Mostrar widgets
display(file_name, data_directory, folder, rois_list, masks_list, mask_noise, slic, folder_ranges, run_button, output)

def run_analysis(b):
    with output:
        print("Data processing started...")
        rois = rois_list.value.split(',')
        masks = list(map(int, masks_list.value.split(',')))
                
        def generar_rangos_discontinuos(rangos_str):
            carpetas = []
            for rango in rangos_str.split(','):
                desde, hasta = map(int, rango.split('-'))
                carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
            return carpetas

        carpetas = generar_rangos_discontinuos(folder_ranges.value)
        carpeta_info = []
        error_carpeta = None

        for carpeta in carpetas:
            try:
                method_path = glob.glob(f"{data_directory.value}/{carpeta}/method")[0]
                param_dict = nogse.nogse_params(method_path)
                x_value = param_dict['ramp_grad_x']
                g_value = param_dict['ramp_grad_str']
                carpeta_info.append((carpeta, x_value, g_value))
            except Exception as e:
                error_carpeta = carpeta
                print(f"Error al procesar la carpeta {carpeta}: {e}")
                break  # Salir del bucle cuando se encuentre el error

        if error_carpeta is not None:
            print(f"El error ocurrió en la carpeta {error_carpeta}.")
            return None, None

        # Ordenar las carpetas primero por x y luego por g
        carpeta_info.sort(key=lambda x: (x[1], x[2]))
        image_paths = []
        method_paths = []

        for carpeta, x_value, g_value in carpeta_info:
            try:
                image_path = glob.glob(f"{data_directory.value}/{carpeta}/pdata/1/2dseq")[0]
                method_path = glob.glob(f"{data_directory.value}/{carpeta}/method")[0]
                image_paths.append(image_path)
                method_paths.append(method_path)
            except Exception as e:
                print(f"Error al procesar la carpeta {carpeta} después de ordenar: {e}")
                break  # Salir del bucle cuando se encuentre el error

        if len(image_paths) == 0 or len(method_paths) == 0:
            print("No se encontraron imágenes o métodos válidos después de ordenar.")
            return None, None
        else:
            print("No se encontraron errores en el procesamiento de las carpetas después de ordenar.")

        fig, ax = plt.subplots(figsize=(8,6)) 

        mask = np.loadtxt(f"scripts/rois/mask_{mask_noise.value}.txt")  
        tnogse, g, n, f_hahn_noise, error_hahn_noise, f_cpmg_noise, error_cpmg_noise, f_A0_hahn_noise, error_A0_hahn_noise, f_A0_cpmg_noise, error_A0_cpmg_noise =  nogse.generate_contrast_vs_g_roi(image_paths, method_paths, mask, slic.value)

        f_hahn_noise = np.array(f_hahn_noise)
        f_cpmg_noise = np.array(f_cpmg_noise)
        f_A0_hahn_noise = np.array(f_A0_hahn_noise)
        f_A0_cpmg_noise = np.array(f_A0_cpmg_noise)

        #print(f_hahn_noise, f_cpmg_noise, f_A0_hahn_noise, f_A0_cpmg_noise)

        f_noise =  (f_cpmg_noise + f_hahn_noise)/2

        palette = sns.color_palette("tab10", len(rois))

        idx = 1
        for roi, color in zip(rois, palette): 

            print("\nROI: ", roi, "\n")

            mask = np.loadtxt(f"scripts/rois/mask_{idx}.txt")

            fig1, ax1 = plt.subplots(figsize=(8,6))

            tnogse, g, n, f_hahn, f_cpmg, f_A0_hahn, f_A0_cpmg, x, TE =  nogse.generate_contrast_vs_g_roi_riciannoise(image_paths, method_paths, mask, slic.value, f_cpmg_noise, f_hahn_noise, f_A0_cpmg_noise, f_A0_hahn_noise)

            print(f"NOGSE parameters for the {len(tnogse)} experiments:\n")
            print("T_nogse:",tnogse)
            print("g:",g)
            print("x:",x)
            print("N:",n)
            print("TE:",TE)

            f_hahn = np.array(f_hahn)
            f_cpmg = np.array(f_cpmg)
            f_A0_hahn = np.array(f_A0_hahn)
            f_A0_cpmg = np.array(f_A0_cpmg)

            f =  f_cpmg/f_A0_cpmg - f_hahn/f_A0_hahn

            directory = f"../nmr-lembo/{file_name.value}/{folder.value}/slice={slic.value}/tnogse={tnogse[0]}_N={int(n[0])}"
            os.makedirs(directory, exist_ok=True)

            nogse.plot_contrast_vs_g_data(ax, roi, g, f, np.abs( ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f) ), tnogse[0], n[0], slic.value, color) 
            nogse.plot_contrast_vs_g_data(ax1, roi, g, f, np.abs( ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f) ), tnogse[0], n[0], slic.value, color)

            table = np.vstack((g, f, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f), f_cpmg, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f_cpmg), f_hahn, ( np.std(f_noise) / np.mean(f_noise) ) * np.array(f_hahn)))
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
        print("Data processing ended...")


run_button.on_click(run_analysis)
