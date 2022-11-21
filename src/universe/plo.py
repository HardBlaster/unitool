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
    TODO
    Generates string indexes. The indexes contain from 1 to 'length' number of sequences. Every sequence is sampled from
    'dictionary'. Every possible variation is generated meaning the length of the returned list is: dict_len^1 + ... +
    dict_len^length.

    :param length: length of the longest index.
    :param dictionary: iterable of characters or strings.
    :return:
    """ 
    # Assume to use with a batch
    assert len(plo_coords.shape) > 2, 'Input shape should be at least 3d, first one is the batch size.'

    # TODO what if there is no batch
    masks = np.zeros((plo_coords.shape[0],) + mask_shape)
    plo_coords_int = np.around(plo_coords).astype(int)
    max_x = int(mask_shape[0] - 1.)
    max_y = int(mask_shape[1] - 1.)
    for image_idx, image_plo_coords in enumerate(plo_coords_int):
        for coords in image_plo_coords:
            if coords[2] == 0.:
                break
            masks[image_idx, min(coords[1], max_y), min(coords[0], max_x)] = 1.

    return masks

# TODO add some disk-shape weight to be more exact-distance like
def point_like_nms(coord_wise_preds: np.ndarray,
                   supression_dist: int = -1) -> np.ndarray:
    
    # For the sake of simplicity we use a box with the point in center, not exact distance for the supression
    
    # we assume here simple 2d data, with shape like (512, 512)
    assert len(coord_wise_preds.shape) == 2, 'Input shape should be 2d.'
    
    # TODO prepare some proper supression distance... for now we use 10th of the width of the image
    if supression_dist == -1:
        supression_dist = (coord_wise_preds.shape[1] // 10)
    
    # Suppression distance must be odd to generate the same size of maxpooling output as input
    supression_dist += (supression_dist % 2) == 0
    # Padding size should reflect the suppression distance
    padding_size = (supression_dist - 1) // 2
    
    preds_reshaped = coord_wise_preds[tf.newaxis, :, :, tf.newaxis]
    res_reshaped = tf.nn.max_pool2d(preds_reshaped,
                                    ksize=(supression_dist, supression_dist),
                                    strides=(1, 1),
                                    padding=[[0, 0],
                                             [padding_size, padding_size],
                                             [padding_size, padding_size],
                                             [0, 0]])
    shape_origi = tf.shape(coord_wise_preds)
    result = tf.reshape(res_reshaped,
                        (shape_origi[0], shape_origi[1]))
    nms_done = tf.where((result == coord_wise_preds),
                        coord_wise_preds,
                        tf.zeros_like(coord_wise_preds, dtype=tf.float32))
    
    # TODO: check threashold, NOT equality, this way we will be able to keep good, but nearby predictions !!
    
    return nms_done 

# visualization for the point-like annotations and predictions
# plo means: Point-like-object
# TODO add param to determine is it a map-like or point-like data
def visualize_plo_mask(images: np.ndarray,
                       masks: np.ndarray,
                       cnt: int = -1,
                       model: tf.keras.Model = None,
                       fade_original: float = .6,
                       use_point_like_nms: bool = False,
                       threshold: float = 0.,
                       keep_max_n_best: int = 0,
                       dilate_interations: int = 2,
                       show_grid_step: int = 0):
    
    if cnt == -1:
        cnt = len(images)
        
    if model:
        pred = model(images)
        
#     # If y is not in map format, covert plo coordinates into map
#     if point_like_target:
#         target = plo.generate_masks_from_plo_coords(target, (input.shape[1], input.shape[2]))
            
    fig, ax = plt.subplots(cnt, 2, figsize=(20, (10 * cnt)))
    for i in range(cnt):

        # Original image
        input_image = images[i]

        target_image = input_image.copy()
        ax[i, 0].imshow((input_image.squeeze()))

        # Ground truth
        target_1d = masks[i]                
        target_1d = cv2.dilate(target_1d, DILATE_MATRIX, iterations=dilate_interations)
        target_image = target_image * fade_original
        target_image[..., 0] = np.clip(target_image[..., 0] + target_1d, 0., 1.)
        
        # Prediction if needed
        if model:
            pred_1d = pred[i].numpy()
            max_n_threshold = 0.
            
            if use_point_like_nms:
                pred_1d = point_like_nms(np.squeeze(pred_1d)).numpy()
            if keep_max_n_best > 0:
                max_n_threshold = np.sort(pred_1d.flat)[-keep_max_n_best - 1]
            if threshold > 0. or max_n_threshold > 0.:
                _, pred_1d = cv2.threshold(pred_1d,
                                           max(threshold, max_n_threshold),
                                           1., cv2.THRESH_BINARY)
                
            pred_1d = cv2.dilate(pred_1d, DILATE_M, iterations=dilate_interations)
            # Add the predictions to the target image as a new layer
            target_image[..., 1] = np.clip(target_image[..., 1] + pred_1d, 0., 1.)

        ax[i, 1].imshow((target_image.squeeze()))

        # Add grid to help the user to see distances
        if show_grid_step > 0:
            ax[i, 1].set_xticks(np.arange(0, target_image.shape[1], show_grid_step))
            ax[i, 1].set_yticks(np.arange(0, target_image.shape[0], show_grid_step))
            ax[i, 1].grid(which='major', color='w', alpha=0.5, linestyle='-', linewidth=1)
