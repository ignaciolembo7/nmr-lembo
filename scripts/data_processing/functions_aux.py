from brukerapi.dataset import Dataset as ds
import numpy as np
import nogse

def extract_images_and_params_with_A0(image_paths, method_paths, slic):
    experiments, A0s, params = [], [], []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:, :, slic, 0])
        experiments.append(ims[:, :, slic, 1])
        param_dict = nogse.nogse_params(method_path)
        params.append(list(param_dict.values()))
    
    return np.array(experiments), np.array(A0s), np.array(params).T

def extract_images_and_params_without_A0(image_paths, method_paths, slic):
    experiments, A0s, params = [], [], []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        experiments.append(ims[:, :, slic, 0])
        param_dict = nogse.nogse_params(method_path)
        params.append(list(param_dict.values()))
    
    return np.array(experiments), np.array(A0s), np.array(params).T

def process_roi(roi_data, mask, noise_data=None, key=None, index=None):
    roi = np.zeros_like(roi_data)
    roi[mask == 255] = roi_data[mask == 255]
    values = roi[roi != 0]
    
    if values.size == 0:
        return 0, 0  # Si no hay datos en la ROI, retornar (valor medio, desviación estándar)

    mean_value = np.mean(values)
    std_dev = np.std(values, ddof=1)  # ddof=1 para el estimador insesgado

    if noise_data is not None and key in noise_data:
        corrected_value = np.sqrt(np.abs(np.mean(values ** 2) - noise_data[key][index] ** 2))
        return corrected_value, std_dev
    
    return mean_value, std_dev

def process_contrast_vs_g_roi_with_A0(image_paths, method_paths, mask, slic, noise_data=None):
    E_matrix, A0_matrix, (tnogse, g, n, x, TE) = extract_images_and_params_with_A0(image_paths, method_paths, slic)
    middle_idx = len(E_matrix) // 2
    
    E_hahn, E_cpmg = E_matrix[:middle_idx], E_matrix[middle_idx:]
    A0_hahn, A0_cpmg = A0_matrix[:middle_idx], A0_matrix[middle_idx:]
    g_contrast = g[:middle_idx]
    
    f_hahn, err_hahn = zip(*[process_roi(E_hahn[i], mask, noise_data, 'hahn', i) for i in range(len(E_hahn))])
    f_cpmg, err_cpmg = zip(*[process_roi(E_cpmg[i], mask, noise_data, 'cpmg', i) for i in range(len(E_cpmg))])
    f_A0_hahn, err_A0_hahn = zip(*[process_roi(A0_hahn[i], mask, noise_data, 'A0_hahn', i) for i in range(len(A0_hahn))])
    f_A0_cpmg, err_A0_cpmg = zip(*[process_roi(A0_cpmg[i], mask, noise_data, 'A0_cpmg', i) for i in range(len(A0_cpmg))])

    f_hahn = np.array(f_hahn)
    f_cpmg = np.array(f_cpmg)
    f_A0_hahn = np.array(f_A0_hahn)
    f_A0_cpmg = np.array(f_A0_cpmg)
    err_hahn = np.array(err_hahn)
    err_cpmg = np.array(err_cpmg)
    err_A0_hahn = np.array(err_A0_hahn)
    err_A0_cpmg = np.array(err_A0_cpmg)
    
    return tnogse, g_contrast, n, x, TE, f_hahn, err_hahn, f_cpmg, err_cpmg, f_A0_hahn, err_A0_hahn, f_A0_cpmg, err_A0_cpmg

def process_nogse_vs_x_roi_with_A0(image_paths, method_paths, mask, slic, noise_data=None):
    E_matrix, A0_matrix, (tnogse, g, n, x, TE) = extract_images_and_params_with_A0(image_paths, method_paths, slic)
    
    f, err_f = zip(*[process_roi(E_matrix[i], mask, noise_data, 'signal', i) for i in range(len(E_matrix))])
    f_A0, err_f_A0 = zip(*[process_roi(A0_matrix[i], mask, noise_data, 'A0', i) for i in range(len(A0_matrix))])

    f = np.array(f)
    f_A0 = np.array(f_A0)
    err_f = np.array(err_f)
    err_f_A0 = np.array(err_f_A0)
    
    return tnogse, g, n, x, TE, f, err_f, f_A0, err_f_A0

############################################################################################################################

def generate_contrast_vs_g_roi_with_A0(image_paths, method_paths, mask, slic):
    return process_contrast_vs_g_roi_with_A0(image_paths, method_paths, mask, slic)

def generate_contrast_vs_g_roi_with_A0_riciannoise(image_paths, method_paths, mask, slic, f_cpmg_noise, f_hahn_noise, f_A0_cpmg_noise, f_A0_hahn_noise):
    noise_data = {'cpmg': f_cpmg_noise, 'hahn': f_hahn_noise, 'A0_cpmg': f_A0_cpmg_noise, 'A0_hahn': f_A0_hahn_noise}
    return process_contrast_vs_g_roi_with_A0(image_paths, method_paths, mask, slic, noise_data)

def generate_NOGSE_vs_x_roi_with_A0(image_paths, method_paths, mask, slic):
    return process_nogse_vs_x_roi_with_A0(image_paths, method_paths, mask, slic)

def generate_NOGSE_vs_x_roi_with_A0_riciannoise(image_paths, method_paths, mask, slic, f_noise, f_A0_noise):
    noise_data = {'signal': f_noise, 'A0': f_A0_noise}
    return process_nogse_vs_x_roi_with_A0(image_paths, method_paths, mask, slic, noise_data)


