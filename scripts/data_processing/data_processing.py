import ipywidgets as widgets
from IPython.display import display
from contrastvsg_processing_with_A0 import contrastvsg_processing_with_A0
from nogsevsx_processing_with_A0 import nogsevsx_processing_with_A0

# Definición de widgets
layout = widgets.Layout(width='90%')
style = {'description_width': 'initial'}

file_name = widgets.Text(value="nmr-lembo/data_processing_fantomas_20230302/nogse_vs_x", description="Save folder:", layout=layout, style=style)
data_directory = widgets.Text(value="C:/Users/ignacio/data/data_fantomas_20230302", description="Data folder:", layout=layout, style=style)
folder_ranges =widgets.Text(value='34-45,47-58', description='Ingrese un conjunto de rangos de carpetas, por ejemplo, 106-108,110-115, ... : ', layout=layout, style=style)
rois_list = widgets.Text(value="Fibra_grande,Fibra_Chica,Agua", description="ROI:", layout=layout, style=style)
mask_noise = widgets.Text(value="4", description="Máscara Ruido:", layout=layout, style=style)
slic = widgets.IntText(value=0, description="Slice:", layout=layout, style=style)
num_grad = widgets.Text(value="1", description="Número Gradiente:", layout=layout, style=style)
analysis_type = widgets.Dropdown(options=['contrast vs g', 'NOGSE vs x'], value=None, description='Tipo de Análisis:', layout=layout, style=style)
run_button = widgets.Button(description="Ejecutar Análisis", layout=layout,style=style)
output = widgets.Output()

# Contenedor de widgets para mantener la estructura
widgets_box = widgets.VBox([file_name, data_directory, rois_list, mask_noise, slic, analysis_type])

# Función para mostrar los widgets correspondientes según el tipo de análisis
def update_widgets(change):
    if analysis_type.value == 'contrast vs g':
        # num_grad.layout.visibility = 'hidden'  # Oculta el widget num_grad
        widgets_box.children = [file_name, data_directory, rois_list, mask_noise, slic, analysis_type, folder_ranges, run_button]  # Mantener solo el botón y demás widgets
    elif analysis_type.value == 'NOGSE vs x':
        num_grad.layout.visibility = 'visible'  # Muestra el widget num_grad
        widgets_box.children = [file_name, data_directory, rois_list, mask_noise, slic, analysis_type, folder_ranges, num_grad, run_button]  # Muestra el num_grad también

# Función de análisis
def run_analysis(b):
    with output:
        output.clear_output()
        print("Iniciando análisis...")

        if analysis_type.value == 'contrast vs g':
            analyzer = contrastvsg_processing_with_A0(
                file_name=file_name.value,
                data_directory=data_directory.value,
                folder_ranges=folder_ranges.value,
                slic=slic.value,
                rois_list=rois_list.value,
                mask_noise=mask_noise.value,
                num_grad=num_grad.value
            )
        else:
            analyzer = nogsevsx_processing_with_A0(
                file_name=file_name.value,
                data_directory=data_directory.value,
                folder_ranges=folder_ranges.value,
                slic=slic.value,
                num_grad=num_grad.value,
                rois_list=rois_list.value,
                mask_noise=mask_noise.value,
            )
        analyzer.load_data()
        analyzer.analyze()

        print("Procesamiento terminado..")

# Mostrar widgets y botón inicialmente
display(widgets_box)

# Conectar el evento de selección de análisis para actualizar los widgets mostrados
analysis_type.observe(update_widgets, names='value')

# Mostrar salida al final, debajo de todo
display(output)
run_button.on_click(run_analysis)
