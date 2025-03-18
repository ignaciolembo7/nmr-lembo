import glob 
import nogse

class data_analyzer:
    def __init__(self, file_name, data_directory, folder_ranges, rois_list, slic, mask_noise, num_grad):
        self.file_name = file_name
        self.data_directory = data_directory
        self.folder_ranges = folder_ranges
        rois_list = rois_list.split(',')
        self.rois_list = rois_list
        self.slic = slic
        self.num_grad = num_grad
        self.mask_noise = mask_noise
        self.image_paths = []
        self.method_paths = []
    
    def load_data(self):
        carpetas = []
        for rango in self.folder_ranges.split(','):
            desde, hasta = map(int, rango.split('-'))
            carpetas.extend([str(numero) for numero in range(desde, hasta + 1)])
    
        carpeta_info = []
        error_carpeta = None
        for carpeta in carpetas:
            try:
                method_path = glob.glob(f"{self.data_directory}/{carpeta}/method")[0]
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

        for carpeta, x_value, g_value in carpeta_info:
            try:
                image_path = glob.glob(f"{self.data_directory}/{carpeta}/pdata/1/2dseq")[0]
                method_path = glob.glob(f"{self.data_directory}/{carpeta}/method")[0]
                self.image_paths.append(image_path)
                self.method_paths.append(method_path)
            except Exception as e:
                print(f"Error al procesar la carpeta {carpeta} después de ordenar: {e}")
                break  # Salir del bucle cuando se encuentre el error

        if len(self.image_paths) == 0 or len(self.method_paths) == 0:
            print("No se encontraron imágenes o métodos válidos después de ordenar.")
            return None, None
        else:
            print("No se encontraron errores en el procesamiento de las carpetas después de ordenar.")

    def analyze(self):
        raise NotImplementedError("Este método debe ser sobrescrito en una subclase")
