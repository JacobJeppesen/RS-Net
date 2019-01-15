from PIL import Image
from src.utils import get_model_name


def get_predicted_thumbnails(file, thresholded, area, transparency, thumbnail_res, params):
    '''
    Used in Jupyter notebooks.
    Load one of the predicted image files from the output library. Return thumbnails of the RGB image and the RGB images
    overlaid with the masks from the Unet, sen2cor, and Fmask algorithms.
    '''
    model_name = get_model_name(params)
    if params.satellite == 'Sentinel-2':
        file = file[0:26]
    elif params.satellite == 'Landsat8':
        file = file[0:21]
    background = Image.open('../data/output/' + file + '-image.tiff').crop(area)
    background.thumbnail(thumbnail_res, Image.NEAREST)

    # Overlay the predicted mask
    if thresholded:
        overlay = Image.open('../data/output/' + file + '_%s.tiff' % (model_name)).crop(
            area)
        overlay = threshold_prediction(overlay, params.threshold)
        overlay.thumbnail(thumbnail_res, Image.NEAREST)
        predicted = overlay_images(background, overlay, transparency)

    else:
        overlay = Image.open('../data/output/' + file + '_%s.tiff' % (model_name)).crop(
            area)
        r = overlay.point(lambda i: i * 253 / 255)
        g = overlay.point(lambda i: i * 231 / 255)
        b = overlay.point(lambda i: i * 36 / 255)
        overlay = Image.merge('RGB', (r, g, b))
        overlay.putalpha(transparency)
        overlay.thumbnail(thumbnail_res, Image.NEAREST)
        background = background.convert('RGBA')
        predicted = Image.alpha_composite(background, overlay)

    if params.satellite == 'Sentinel-2':
        # Overlay the sen2cor and fmask masks
        overlay_sen2cor_org = Image.open('../data/output/' + file + '_cls-%s_sen2cor.tiff' % params.cls).crop(area)
        overlay_sen2cor_org.thumbnail(thumbnail_res, Image.NEAREST)
        predicted_sen2cor = overlay_images(background, overlay_sen2cor_org, transparency)

        overlay_fmask_org = Image.open('../data/output/' + file + '_cls-%s_fmask.tiff' % params.cls).crop(area)
        overlay_fmask_org.thumbnail(thumbnail_res, Image.NEAREST)
        predicted_fmask = overlay_images(background, overlay_fmask_org, transparency)

        return background, predicted, predicted_sen2cor, predicted_fmask

    elif params.satellite == 'Landsat8':
        overlay_true = Image.open('../data/output/' + file + '_true_cls-%s_collapse%s.tiff' % ("".join(str(c) for c in params.cls), params.collapse_cls)).crop(area)
        overlay_true.thumbnail(thumbnail_res, Image.NEAREST)
        mask_true = overlay_images(background, overlay_true, transparency)

        return background, predicted, mask_true


def threshold_prediction(prediction, threshold):
    '''
    Thresholds a saliency map and returns a binary mask
    '''
    # See https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent
    # and https://stackoverflow.com/questions/10640114/overlay-two-same-sized-images-in-python
    prediction = prediction.convert("RGBA")
    datas = prediction.getdata()
    newData = []
    for item in datas:
        if item[0] >= threshold * 255:
            newData.append((253, 231, 36, 255))  # The pixel values for '1' in the mask
        else:
            newData.append((0, 0, 0, 0))  # Makes it completely transparent
    prediction.putdata(newData)

    return prediction


def overlay_images(background, overlay, transparency):
    '''
    Overlay a mask on an RGB image
    '''
    # See https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent
    # and https://stackoverflow.com/questions/10640114/overlay-two-same-sized-images-in-python
    overlay = overlay.convert("RGBA")
    background = background.convert("RGBA")
    datas = overlay.getdata()
    newData = []
    for item in datas:
        if item[0] == 0:
            newData.append((0, 0, 0, 0))  # Makes it completely transparent
        else:
            newData.append((255, 120, 0, transparency))  # The pixel values for '1' in the mask
    overlay.putdata(newData)

    return Image.alpha_composite(background, overlay)