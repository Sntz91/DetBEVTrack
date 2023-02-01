import numpy as np

def get_mask_if_channel_low(img, channel, threshold):
    """
        Return Mask of size (img.h, img.w) where channel is below 
        certain threshold.
    """
    assert channel in [0,1,2]
    channel = img[:,:,channel]
    mask = channel < threshold
    return mask

def remove_background(image, mask):
    """
        Remove Background of image, given mask of size (img.h, img.w) 
    """
    assert image.shape[0:2] == mask.shape
    image = add_alpha_channel(image)
    image[:,:,3][mask] = 0
    return image

def add_alpha_channel(image):
    """
        Add alpha channel to an image, with 255 each.
    """
    alpha_channel = (np.ones(image.shape[0:2])*255).astype(np.uint8)
    new_image = np.dstack((image, alpha_channel))
    return new_image
