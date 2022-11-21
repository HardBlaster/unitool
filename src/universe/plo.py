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
    Generates plo mask matricies from object coordinates. Output masks will be 0. everywhere, except the given coordiantes,
    where the output values will be 1.
    
    We assume input contains all batches, so the assumed shape is liket: (batch_cnt, object_cnt, 3), batch_cnt is the
    number of images in the batch, object_cnt is the maximum number of objects on one image. The last dimension is 3,
    here one line defines one object with this format, all values must be float: [y, x, 1.]. The last value is 1.,
    if there is an object defined on that line. If there are no more objects, all remaining lines should be [0., 0., 0.].

    :param plo_coords: Point-like object coordinates in the format defined above.
    :param mask_shape: Shape of the output matricies, like (height, width).
    :return: The generated mask matricies with the shape: (batch_cnt, height, width). Values are 1. where objects are defined,
    otherwise zeros.
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
def point_like_nms(plo_mask_preds: np.ndarray,
                   supression_dist: int = -1) -> np.ndarray:
    """
    Non-max suppression function written in tensorflow for point-like object masks.
    
    For now the distance is approximated with a simple square with the actual point in the middle.
    ( Later a circural weight must be added to calculate with real distances )

    :param plo_mask_preds: Predicted plo mask, values assumed to be logits, float numbers in 0..1, and shape to be like (height, width)
    :param supression_dist: Supression distance to use, this is the actual pixel distance. It must be an odd number
    (but the code handles it), default value is 10 % of the original width, if supression_dist is -1.
    :return: The supressed mask. Supressed values become zeros, other values keep their original values.
    """ 
    
    # For the sake of simplicity we use a box with the point in center, not exact distance for the supression
    
    # we assume here simple 2d data, with shape like (512, 512)
    assert len(plo_mask_preds.shape) == 2, 'Input shape should be 2d.'
    
    # TODO prepare some proper supression distance... for now we use 10th of the width of the image
    if supression_dist == -1:
        supression_dist = (plo_mask_preds.shape[1] // 10)
    
    # Suppression distance must be odd to generate the same size of maxpooling output as input
    supression_dist += (supression_dist % 2) == 0
    # Padding size should reflect the suppression distance
    padding_size = (supression_dist - 1) // 2
    
    preds_reshaped = plo_mask_preds[tf.newaxis, :, :, tf.newaxis]
    res_reshaped = tf.nn.max_pool2d(preds_reshaped,
                                    ksize=(supression_dist, supression_dist),
                                    strides=(1, 1),
                                    padding=[[0, 0],
                                             [padding_size, padding_size],
                                             [padding_size, padding_size],
                                             [0, 0]])
    shape_origi = tf.shape(plo_mask_preds)
    result = tf.reshape(res_reshaped,
                        (shape_origi[0], shape_origi[1]))
    plo_nms = tf.where((result == plo_mask_preds),
                       plo_mask_preds,
                       tf.zeros_like(plo_mask_preds, dtype=tf.float32))
    
    # TODO: check threashold, NOT equality, this way we will be able to keep good, but nearby predictions !!
    
    return plo_nms 

# Plo nms related code snippets, copy somewhere and run !
# # #
# To understand the point like non max supression, here is an example
# frt = tf.constant([[1, 2, 3, 8],
#                    [7, 8, 9, 3],
#                    [4, 5, 3, 2],
#                    [7, 2, 2, 9]])
# frt_r = frt[tf.newaxis, :, :, tf.newaxis]
# result = tf.nn.max_pool2d(frt_r, ksize=(3, 3),
#                           strides=(1, 1),
#                           padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
# result_o = tf.reshape(result, (4, 4))
# tf.where((result_o == frt),
#          frt,
#          tf.zeros_like(frt, dtype=tf.int32))
# # #
# To meausure how fast is the point like non max supression
# alap = np.random.rand(10240, 10240)
# nms = plo.point_like_nms(alap, supression_dist=200)
# y, idx, count = tf.unique_with_counts(tf.reshape(nms, -1))
# print(y, count)


# visualization for the point-like annotations and predictions
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
    """
    Visualization for point-like object masks using matplotlib.
    Print out samples, one sample by row. In each row there are two images, on the left the original image.
    On the right the image faded out, and the point-like objects added on top of it. The defined masks are added
    in red color, dilated up by number defined in the dilate_interations param. If model is defined,
    the predicted objects are added in green color. Therefore the matching points will be yellow.

    :param images: The original images with the shape of (batch_cnt, height, width, channerls)
    :param masks: The plo masks (usually the ground truth), the shape assumed like (batch_cnt, height, width)
    :param cnt: How many images should be visulazied, the first cnt number of images will be used from the batch.
    Default value is -1, which means all images will be visualized from the given batch.
    :param model: The model to be used to predict plo masks. If not given, predictions won't be added.
    :param fade_original: Float multiplier, how much the original image should be faded out in the second column.
    Default value is 0.6, this keeps the image still visible, but also gives more visibility to the object points.
    :param use_point_like_nms: Use point-like non-max supression, or not. Default value is False.
    :param threashold: Threshold to cut down unwanted predictons. Only used if there is a model given.
    All values below becomes to zero, all value equals or above becomes 1. Default value is 0.
    :param keep_max_n_best: Keep only the best n predictions. Only used if there is a model given. This means
    that it will keep only the best predictions, all other will be supressed. If used together with threashold,
    the given threashold will be applied after, so it can further supress values from the predictions.
    Default value is 0, which means all predictions are kept.
    :param dilate_interations: To have a better visual of point-like  objects, points can be dilated up, by
    the number of iterations given here in this parameter. Default value is 2 which is an optimal solution for
    dimensions around 500.
    :param show_grid_step: Int value, add grid to the image, using this step number given here used as pixels.
    Default value is 0, which means no grid will be added to the images in the second column.
    :return:
    """  
    
    if cnt == -1:
        cnt = len(images)
        
    if model:
        pred = model(images)
    
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
                
            pred_1d = cv2.dilate(pred_1d, DILATE_MATRIX, iterations=dilate_interations)
            # Add the predictions to the target image as a new layer
            target_image[..., 1] = np.clip(target_image[..., 1] + pred_1d, 0., 1.)

        ax[i, 1].imshow((target_image.squeeze()))

        # Add grid to help the user to see distances
        if show_grid_step > 0:
            ax[i, 1].set_xticks(np.arange(0, target_image.shape[1], show_grid_step))
            ax[i, 1].set_yticks(np.arange(0, target_image.shape[0], show_grid_step))
            ax[i, 1].grid(which='major', color='w', alpha=0.5, linestyle='-', linewidth=1)

