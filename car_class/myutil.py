#!/root/anaconda2/bin/python
from collections import defaultdict
import numpy as np
import PIL.Image as pil_image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from keras import backend as K

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               thickness=4,
                               use_normalized_coordinates = True
                              ):
    
    """Adds a bounding box to an image.
      Each string in display_str_list is displayed on a separate line above the
      bounding box in black text on a rectangle filled with the input 'color'.
      Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                          (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
          ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
          coordinates as absolute.
    """
    
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax) 
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill='red')

def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.
      Args:
        image: a PIL.Image object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
               The coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
                               a list of strings for each bounding box.
                               The reason to pass a list of strings for a
                               bounding box is that it might contain
                               multiple labels.
      Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3])
    
def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def original(img):
    tmp = np.zeros_like(img)
    for i in range(3):
        tmp[:,:,i] = img[:,:,i] - img[:,:,i].min()
    return tmp.astype(np.uint8)

def lookup(d, key, i):
    if key == 'make+model':
        for item in d:
            if item['pp_brand_id']+' '+item['pp_genre_id'] == i:
                return item['chinese']
    else:
        for item in d:
            if item[key] == i:
                return item['chinese']
            
def crop_img(img, box):
    """
    box: left, top, right, bottom
    """
    w, h = img.size
    box = [box[0]*w, box[1]*h,box[2]*w,box[3]*h]
    return img.crop(box)
            
def load_img(path, grayscale=False, target_size=None, box=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        box: The crop rectangle, as a (left, upper, right, lower)-tuple.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if box:
        img = crop_img(img, box)
        
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        #resize(w, h)
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img
            
def load_data(image_paths, labels, num_of_class=10, target_size=(227, 227), box=None):
    """
    Given list of paths, resize and bounding box load images as one numpy array of shape
        (num_images, crop_size, crop_size, channel)
        box:[top, left, bottom, right]
    :return X: image array
     return y: one hot encoded labels
    """
    if box:
        X = np.zeros((len(image_paths), target_size[0],target_size[1], 3))
        ## google output box :## 0: top 1: left 2 lower 3 right
        for i,path in enumerate(image_paths):
            new_box = (box[i][1],box[i][0],box[i][3], box[i][2])
            X[i, :] = img_to_array(load_img(path, target_size=target_size, box=new_box))
        y = np_utils.to_categorical(labels, num_of_class)
        return X, y
    else:
        X = np.zeros((len(image_paths), target_size[0],target_size[1], 3))
        for i,path in enumerate(image_paths):
            X[i, :] = img_to_array(load_img(path, target_size=target_size))
        y = np_utils.to_categorical(labels, num_of_class)
        return X, y
    
def load_data_flip(image_paths, labels, num_of_class=10, target_size=(227, 227), box=None):
    """
    Given list of paths, resize and bounding box load images as one numpy array of shape
        (num_images, crop_size, crop_size, channel)
        box:[top, left, bottom, right]
    :return X: image array
     return y: one hot encoded labels
    """
    if box.any():
        X = np.zeros((len(image_paths), target_size[0],target_size[1], 3))
        ## google output box :## 0: top 1: left 2 lower 3 right
        for i,path in enumerate(image_paths):
            new_box = (box[i][1],box[i][0],box[i][3], box[i][2])
            if bool(random.getrandbits(1)):
                X[i, :] = img_to_array(load_img(path, target_size=target_size, box=new_box))
            else: 
                X[i, :] = img_to_array(load_img(path, target_size=target_size, box=new_box))[:,::-1,:]
        y = np_utils.to_categorical(labels, num_of_class)
        return X, y
    else:
        X = np.zeros((len(image_paths), target_size[0],target_size[1], 3))
        for i,path in enumerate(image_paths):
            if bool(random.getrandbits(1)):
                X[i, :] = img_to_array(load_img(path, target_size=target_size))
            else: 
                X[i, :] = img_to_array(load_img(path, target_size=target_size))[:,::-1,:]
        y = np_utils.to_categorical(labels, num_of_class)
        return X, y
    
def judge_box(left, right, left_t, right_t):
    if left < left_t and right > right_t:
        return True

def load_box(path,x_threshold=0.51,x_threshold2=0.51):
    if x_threshold > x_threshold2:
        print 'threshold error'
    with open(path) as f:
        bb_list = np.load(f).item()
    result = defaultdict(list)
    ## delete left > 0.5 right border < 0.5
    ## 0: top 1: left 2 lower 3 right
    for k in bb_list.keys():
        if len(bb_list[k]) == 1:
            result[k] = bb_list[k]
            continue
        for bb_box in bb_list[k]:
            if judge_box(left=bb_box[1], right=bb_box[3], left_t=x_threshold, right_t=x_threshold2):
                result[k].append(bb_box)
    return result


