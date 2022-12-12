from typing import Dict, List, Tuple
import numpy as np
import cv2

def image_and_plo_augmentation(image: np.ndarray,
                               label: np.ndarray,
                               labels_are_coords: bool = True,
                               rotate_angle: float = 10.,
                               zoom_scale: float =.2,
                               shear_factor: float =.1,
                               bg_color: Tuple[int, int, int] = (255, 255, 255),
                               drop_if_coord_off: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Image augmentation function to handle both the image and the special plo formatted array,
    but can handle simple label masks too. The augmentations are the following it can carry out:
    - centered rotation
    - centered scaling (zooming in / out)
    - shearing
    - horizontal flip
    
    If you use the special plo format as label output you can avoid augmenting objects off the image by forcing
    to recalculate it until all objects remain inside the image. The function decreases all augmentation values
    during the recalculation cycles to avoid running into an infinite loop.
    
    :param image: Image, shape should be like (height, width, channels)
    :param label: Labels for image, it can be a simple label mask or in plo format.
    :param labels_are_coords: Boolean value indicating whether the input label is in plo or mask format.
    :param rotate_scale: Max rotation to apply, in angles. Default value is 10 degrees. 
    :param zoom_scale: Max zoom in / out to apply. Default value is 0.2 .
    :param shear_factor: Max shear to apply. Default value is 0.1 .
    :param bg_color: Background color to use off-image pixels becomes visible after the transformation, in the format of (red, green, blue),
    int values within the range 0..255. Default value is (255, 255, 255), meaning a white background color.
    :return: Returns the augmented image and labels.
    """
    
    height, width = image.shape[0], image.shape[1]
    redo_trans = True
    redo_factor = 1.
    redo_mul = .4
    
    # If labels are in coordinate list format, then it first must be converted  into the proper matrix format
    if labels_are_coords:
        # Create the helper array according to x coordinate. 0. if x is -1, otherwise use 1.
        add_column =  np.array([0. if at == -1. else 1. for at in label[:, 0]])
        # Add the new array as a column
        label = np.hstack((label, add_column[:, np.newaxis]))
        # Switch -1. values to 0.
        label = np.where(label == -1., 0., label)
    
    while redo_trans:
        
         # Redo needed only if working with coords AND we should avoid off-image coordinates
        redo_trans = False
        
        # Shear, centered
        shear_mat = np.float32([[1., np.random.uniform(-shear_factor, shear_factor) * redo_factor, 0.],
                                [np.random.uniform(-shear_factor, shear_factor) * redo_factor, 1., 0.]])
        # Here we use the same original center point as the center for the shear transformations
        shear_mat[0, 2] = -shear_mat[0, 1] * (height / 2)
        shear_mat[1, 2] = -shear_mat[1, 0] * (width / 2)

        # Rotate and scale, both centered
        center_x, center_y = width // 2, height // 2
        rotate_mat = cv2.getRotationMatrix2D((center_x, center_y),
                                             np.random.uniform(-rotate_angle, rotate_angle) * redo_factor,
                                             (np.random.uniform(0., zoom_scale) * redo_factor) + 1.0)      
        if labels_are_coords:
            # Apply the transformations on the label coordinates
            calculated_label = np.dot(shear_mat, np.transpose(label))
            trans_label = np.column_stack((np.transpose(calculated_label), label[:, 2]))
            calculated_label = np.dot(rotate_mat, np.transpose(trans_label))
            trans_label = np.column_stack((np.transpose(calculated_label), trans_label[:, 2]))  
            
            # Recalc the transformation matrices IF any transformed coordinate is off-image
            # AND IF user indicated to do that
            if drop_if_coord_off:
                if np.any(trans_label < -.5) \
                    | np.any(trans_label[:, 0] > (width - .5)) \
                    | np.any(trans_label[:, 1] > (height - .5)):
                    redo_trans = True
                    redo_factor *= redo_mul
    
    # Apply transformations to the image
    trans_image = cv2.warpAffine(image, shear_mat, (height, width), borderValue=bg_color)
    trans_image = cv2.warpAffine(trans_image, rotate_mat, (height, width), borderValue=bg_color)
    
    # If labels in coordinates they must be clipped into the valid range 0. .. (dim - 1.)
    if labels_are_coords:
        trans_label[:, 0] = np.clip(trans_label[:, 0], 0., width - 1.)
        trans_label[:, 1] = np.clip(trans_label[:, 1], 0., height - 1.)
    else:
        # Else the transformations should be applied the same way as to the image
        trans_label = cv2.warpAffine(label, shear_mat, (height, width))
        trans_label = cv2.warpAffine(trans_label, rotate_mat, (height, width))
        
    # Randomly flip frame
    flip = np.random.randint(0, 1)
    if flip:
        trans_image = trans_image[:, ::-1, :]
        if labels_are_coords:
            trans_label[:, 0] = (float(size - 1.) - trans_label[:, 0]) * trans_label[:, 2]
        else:
            trans_label = trans_label[:, ::-1]
    
    # Label coordinates converted back into the simple coordinates list format
    if labels_are_coords:
        # Removing the last column, setting back unused coordintes into -1.
        subtr_column = (1. - trans_label[..., 2])
        trans_label[..., 0] -= subtr_column
        trans_label[..., 1] -= subtr_column
        trans_label = trans_label[..., 0:2]

    return trans_image, trans_label

def color_space_augmentation(image: np.ndarray,
                             h_gain: float = .015,
                             s_gain: float = .7,
                             v_gain: float = .4) -> np.ndarray:
    """
    Image color augmentation in the hsv color space. Input must be in RGB format.
        
    :param image: Image, shape should be like (height, width, channels)
    :param h_gain: Max gain applied on hue.
    :param s_gain: Max gain applied on saturation.
    :param v_gain: Max gain applied on value.
    :return: Returns the augmented image.
    """
    # Random values generated in -1..+1 range, and then moved into the 0..+2
    # to used as a gain value
    r = np.random.uniform(-1., 1., 3) * [h_gain, s_gain, v_gain] + 1.
    h_origi, s_origi, v_origi = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype
    
    x = np.arange(0, 256, dtype=r.dtype)
    h_lut = ((x * r[0]) % 180.).astype(dtype)
    s_lut = np.clip(x * r[1], 0., 255.).astype(dtype)
    v_lut = np.clip(x * r[2], 0., 255.).astype(dtype)
    image_hsv = cv2.merge((cv2.LUT(h_origi, h_lut),
                           cv2.LUT(s_origi, s_lut),
                           cv2.LUT(v_origi, v_lut)))
    
    trans_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    
    return trans_image
