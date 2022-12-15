import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

DILATE_MATRIX: np.ndarray = np.array([[0, 1, 2, 1, 0],
                                      [1, 3, 5, 3, 1],
                                      [2, 5, 5, 5, 2],
                                      [1, 3, 5, 3, 1],
                                      [0, 1, 2, 1, 0]],
                                     dtype=np.uint8)

def generate_masks_from_plo_coords(plo_coords: np.ndarray,
                                   mask_shape: tuple) -> np.ndarray:
    """
    Generates plo mask matrices from object coordinates. Output masks will be 0. everywhere, except the given coordiantes,
    where the output values will be 1.
    
    We assume input contains all batches, so the assumed shape is like: (batch_cnt, object_cnt, 2), batch_cnt is the
    number of images in the batch, object_cnt is the maximum number of objects on one image. The last dimension is 2,
    here one line defines one object with this format, all values must be float: [x, y]. If there are no more objects,
    all remaining lines should be [-1., -1.].
    
    :param plo_coords: Point-like object coordinates in the format defined above.
    :param mask_shape: Shape of the output matrices, like (height, width).
    :return: The generated mask matrices with the shape: (batch_cnt, height, width). Values are 1. where objects are defined,
    otherwise zeros.
    
    Examples
    --------
    # Example point-like object matrix:
    # We have maximum 6 objects, 3 presents on this image.
    # Coordinates must be in the following format: [horizontal_coord, vertical_coord], indexed from 0.
    >>> plom = np.array([[12., 23.],
                        [34., 45.],
                        [56., 67.],
                        [-1., -1.],
                        [-1., -1.],
                        [-1., -1.]])
    """ 
    
    # TODO rewrite in torch, use sparse matrix constructor
    
    # Assume to use with a batch
    assert len(plo_coords.shape) >= 2, 'Input shape should be at least 3d (2d in case of lacking th batch dimension).'
    
    # if no batch dimension found simply add it
    if len(plo_coords.shape) == 2:
        plo_coords = plo_coords[np.newaxis, :]
        
    masks = np.zeros((plo_coords.shape[0],) + mask_shape)
    plo_coords_int = np.around(plo_coords).astype(int)
    max_x = mask_shape[0] - 1
    max_y = mask_shape[1] - 1
    for image_idx, image_plo_coords in enumerate(plo_coords_int):
        for coords in image_plo_coords:
            if coords[0] == -1:
                break
            masks[image_idx, min(coords[1], max_y), min(coords[0], max_x)] = 1.

    return masks

def point_like_nms(mask_preds: np.ndarray,
                   suppression_dist: int = -1) -> np.ndarray:
    """
    Non-max suppression function written in tensorflow for point-like object masks.
    
    For now the distance is approximated with a simple square with the actual point in the middle.
    ( Later a circular weight must be added to calculate with real distances )

    :param mask_preds: Predicted plo mask, values assumed to be logits, float numbers in 0..1, and shape to be like (height, width)
    :param suppression_dist: Suppression distance to use, this is the actual pixel distance. It must be an odd number
    (but the code handles it), default value is 10 % of the original width, if suppression_dist is -1.
    :return: The suppressed mask. Suppressed values become zeros, other values keep their original values.
    
    Examples
    --------
    # This helps to understand the point like non max suppression:
    >>> m = tf.constant([[1, 2, 3, 8],
                        [7, 8, 9, 3],
                        [4, 5, 3, 2],
                        [7, 2, 2, 9]])
    >>> m_r = m[tf.newaxis, :, :, tf.newaxis]
    >>> res = tf.nn.max_pool2d(m_r, ksize=(3, 3), strides=(1, 1), padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
    >>> res_o = tf.reshape(res, (4, 4))
    >>> tf.where((res_o == m), m, tf.zeros_like(m, dtype=tf.int32))
    <tf.Tensor: shape=(4, 4), dtype=int32, numpy=
    array([[0, 0, 0, 0],
           [0, 0, 9, 0],
           [0, 0, 0, 0],
           [7, 0, 0, 9]], dtype=int32)>
    """ 
    
    # For the sake of simplicity we use a box with the point in center, not exact distance for the suppression
    # TODO add some disk-shape weight to be more exact-distance like

    # we assume here simple 2d data, with shape like (512, 512)
    assert len(mask_preds.shape) == 2, 'Input shape should be 2d.'
    
    # TODO prepare some proper suppression distance... for now we use 10th of the width of the image
    if suppression_dist == -1:
        suppression_dist = (mask_preds.shape[1] // 10)
    
    # Suppression distance must be odd to generate the same size of maxpooling output as input
    # TODO rewrite to be more self-explanetory
    suppression_dist += 1 - (suppression_dist % 2)
    # Padding size should reflect the suppression distance
    padding_size = (suppression_dist - 1) // 2
    
    preds_reshaped = mask_preds[tf.newaxis, :, :, tf.newaxis]
    res_reshaped = tf.nn.max_pool2d(preds_reshaped,
                                    ksize=(suppression_dist, suppression_dist),
                                    strides=(1, 1),
                                    padding=[[0, 0],
                                             [padding_size, padding_size],
                                             [padding_size, padding_size],
                                             [0, 0]])
    shape_origi = tf.shape(mask_preds)
    result = tf.reshape(res_reshaped,
                        (shape_origi[0], shape_origi[1]))
    plo_nms = tf.where((result == mask_preds),
                       mask_preds,
                       tf.zeros_like(mask_preds, dtype=tf.float32))
    
    # TODO: check threashold, NOT equality, this way we will be able to keep good, but nearby predictions !!
    
    return plo_nms 

def filter_plo_preds(pred_masks: np.ndarray,
                     plo_nms_suppression_dist: int = 0,
                     threshold: float = 0.,
                     keep_max_n_best: int = 0) -> np.ndarray:
    """
    Filter predicted plo masks according to the given parameters.
    
    :param plo_nms_suppression_dist: Use the point-like non-max suppression with the given distance in pixels.
    If it is 0, plo nms will not be called. If it is -1, plo nms will be called with default distance.
    Default value is 0.
    :param threshold: Threshold to cut down unwanted predictions. Only used if there is a model given.
    All values below become zero, all value equals or above become 1. Default value is 0.
    :param keep_max_n_best: Keep only the best n predictions. Only used if there is a model given. This means
    that it will keep only the best predictions, all others will be suppressed. If used together with threshold,
    the given threshold will be applied after, so it can further suppress values from the predictions.
    Default value is 0, which means all predictions are kept.
    :return: The filtered out array of masks. Shape is the same as the input.
    """
    # Assume to use with a batch
    assert len(pred_masks.shape) > 2, 'Input shape should be at least 3d, first one is the batch size.'
    
    filtered_plo_list = []
    
    for pred in pred_masks.numpy():
        max_n_threshold = 0.

        if plo_nms_suppression_dist != 0:
            pred = point_like_nms(np.squeeze(pred),
                                  suppression_dist=plo_nms_suppression_dist).numpy()
        if keep_max_n_best > 0:
            max_n_threshold = np.sort(pred.flat)[-keep_max_n_best - 1]
        if threshold > 0. or max_n_threshold > 0.:
            _, pred = cv2.threshold(pred,
                                    max(threshold, max_n_threshold),
                                    1., cv2.THRESH_BINARY)
        filtered_plo_list.append(pred)

    return np.array(filtered_plo_list)

def visualize_plo_mask(images: np.ndarray,
                       gt_masks: np.ndarray,
                       pred_masks: np.ndarray = None,
                       fade_original: float = .6,
                       dilate_interations: int = 2,
                       show_grid_step: int = 0):
    """
    Visualization for point-like object masks using matplotlib.
    Print out samples, one sample by row. In each row there are two images, on the left the original image.
    On the right the image faded out, and the point-like objects added on top of it. The defined masks are added
    in red color, dilated up by the number defined in the dilate_interations param. If prediction is defined,
    it will be added in green color. Therefore the matching points will be yellow.

    :param images: The original images with the shape of (batch_cnt, height, width, channels)
    :param gt_masks: The plo masks (usually the ground truth), the shape assumed like (batch_cnt, height, width)
    :param pred_masks: The predicted masks (usually the prediction), the shape must be the same as gt_masks shape.
    :param fade_original: Float multiplier, how much the original image should be faded out in the second column.
    Default value is 0.6, this keeps the image still visible, but also gives more visibility to the object points.
    :param dilate_interations: To have a better visual of point-like  objects, points can be dilated up, by
    the number of iterations given here in this parameter. Default value is 2 which is an optimal solution for
    dimensions around 500.
    :param show_grid_step: Int value, add grid to the image, using this step number given here used as pixels.
    Default value is 0, which means no grid will be added to the images in the second column.
    :return:
    """
        
    cnt = len(images)
    
    # assert, all imags, masks should have the same dimensions
    
    fig, ax = plt.subplots(cnt, 2, figsize=(20, (10 * cnt)))
    for i in range(cnt):

        # Original image
        input_image = images[i]

        target_image = input_image.copy()
        ax[i, 0].imshow((input_image.squeeze()))

        # Ground truth
        target_1d = gt_masks[i]                
        target_1d = cv2.dilate(target_1d, DILATE_MATRIX, iterations=dilate_interations)
        target_image = target_image * fade_original
        target_image[..., 0] = np.clip(target_image[..., 0] + target_1d, 0., 1.)
        
        # Prediction if needed
        if np.any(pred_masks):
            pred_1d = pred_masks[i]                
            pred_1d = cv2.dilate(pred_1d, DILATE_MATRIX, iterations=dilate_interations)
            target_image[..., 1] = np.clip(target_image[..., 1] + pred_1d, 0., 1.)

        ax[i, 1].imshow((target_image.squeeze()))

        # Add grid to help the user to see distances
        if show_grid_step > 0:
            ax[i, 1].set_xticks(np.arange(0, target_image.shape[1], show_grid_step))
            ax[i, 1].set_yticks(np.arange(0, target_image.shape[0], show_grid_step))
            ax[i, 1].grid(which='major', color='w', alpha=0.5, linestyle='-', linewidth=1)

