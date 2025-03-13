#NMRSI - Ignacio Lembo Ferrari - 25/04/2024

import cv2
import numpy as np
from brukerapi.dataset import Dataset as ds

def draw_circle(event, x, y, flags, param):
    global center, radius, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        center = (x, y)
        radius = 0  # Inicializar el radio con un valor predeterminado

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))
        cv2.circle(im_scaled, center, radius, (255), 2)
        cv2.circle(mask, center, radius, (255), 2)
        cv2.floodFill(mask, None, (0, 0), (255))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(im_scaled, center, radius, (255), 2)
            cv2.circle(mask, center, radius, (255), 2)
            current_former_x = x
            current_former_y = y

def draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
                cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
                current_former_x = former_x
                current_former_y = former_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.line(im_scaled, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
            cv2.line(mask, (current_former_x, current_former_y), (former_x, former_y), (255), 2)
            current_former_x = former_x
            current_former_y = former_y
            # Fill the enclosed area in the mask with white
            cv2.floodFill(mask, None, (0, 0), (255))

    return former_x, former_y

serial = input("Serial:") #ms
nrois = 4 #input("Nrois:") #ms
slic = 0
scaling_factor = 4 # Factor de escala (puedes ajustarlo según sea necesario)
file_name = "fantomas_20230302"
ims = ds(f"C:/Users/ignacio/data/data_{file_name}/"+str(serial)+"/pdata/1/2dseq").data

A0_matrix = ims[:,:,slic,0]
M_matrix = ims[:,:,slic,1]
original = A0_matrix

np.savetxt(f"scripts/rois/original.txt", original, fmt='%f')

#Equalize original
original_eq = 255 * (original - np.min(original)) / (np.max(original) - np.min(original)) + 255*(np.min(original) / (np.max(original) - np.min(original)) )

cv2.imwrite(f"rois/original_eq.jpg", original_eq)

im = cv2.imread(f"rois/original_eq.jpg", cv2.IMREAD_GRAYSCALE)

for i in range(nrois):

    drawing = False  # True if mouse is pressed
    mode = True  # If True, draw rectangle. Press 'm' to toggle to curve

    # Escalar la imagen para que se vea más grande
    im_scaled = cv2.resize(im, None, fx=scaling_factor, fy=scaling_factor)
    mask = np.zeros_like(im_scaled)  # Create a black image with the same size as im
    
    cv2.namedWindow("Roi_Select")
    cv2.setMouseCallback('Roi_Select', draw_circle)

    while True:
        cv2.imshow('Roi_Select', im_scaled)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:  # Press Enter to move to the next ROI
            break

    # Invert the mask
    mask_inverted = cv2.bitwise_not(mask)
    
    # Resize the inverted mask to match the size of the original image
    mask_resized = cv2.resize(mask_inverted, (original.shape[1], original.shape[0]))
    mask_resized[mask_resized != 0] = 255
        
    # Apply the mask to the original image
    roi = np.zeros_like(original)
    roi[mask_resized == 255] = original[mask_resized == 255]
    signal = np.mean(roi[roi != 0])
    signal_err = np.std(roi[roi != 0])
    print(f"Average intensity of ROI {i+1}: {signal}")
    print(f"Average intensity error of ROI {i+1}: {signal_err}")

    # Save roi
    np.savetxt(f"rois/roi_{i+1}.txt", roi, fmt='%f')
    #roi = (roi * 255).astype(np.uint8)
    roi_eq = 255 * (roi - np.min(roi)) / (np.max(roi) - np.min(roi)) + 255*(np.min(roi) / (np.max(roi) - np.min(roi)) )
    cv2.imwrite(f"rois/roi_{i+1}.jpg", roi)

    # Save mask
    np.savetxt(f"rois/mask_{i+1}.txt", mask_resized, fmt='%f')
    cv2.imwrite(f"rois/mask_{i+1}.jpg", mask_resized)

cv2.destroyAllWindows()

# Copiar la imagen original para la imagen final
imagen_final = im.copy()
imagen_color = cv2.cvtColor(imagen_final, cv2.COLOR_GRAY2BGR)

for i in range(1, nrois + 1):
    # Leer la máscara de la ROI en escala de grises
    mask_roi = cv2.imread(f"rois/mask_{i}.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Superponer la máscara en la imagen final
    imagen_final = cv2.add(imagen_final, mask_roi)
    cv2.imwrite(f"rois/im={serial}_rois_final.jpg", imagen_final)
    
    # Colorear la ROI en rojo en la imagen color
    imagen_color[mask_roi >= 240] = [0, 0, 255]

# Guardar la imagen final en color con las ROIs superpuestas en rojo
cv2.imwrite(f"../results_{file_name}/images/im={serial}_rois_final_color.jpg", imagen_color)