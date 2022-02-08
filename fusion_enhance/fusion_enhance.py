import math
import time

import cv2
import numpy as np

#########################################################
# the standard filter mask
#########################################################
_h = np.array([1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0])
_h = np.expand_dims(_h, axis=1)
FILTER_MASK = np.matmul(_h, _h.T)

#########################################################
# For debugging and computing timings
#########################################################
_prev_time = None


def _debug_timing(name="Test"):
    global _prev_time
    if _prev_time is not None:
        print(f"Timing {name}: {(time.perf_counter() - _prev_time):.4f} secs")
    _prev_time = time.perf_counter()


#########################################################
# The fusion enhance utility
#########################################################
def enhance(img, level: int = 3, color_balance_clip_percentile: int = 5, clahe_clip_limit: float = 2.0):
    # _debug_timing()
    img = img_to_float32(img)
    # _debug_timing("Float Conversion")
    # color balance
    img1 = SimplestColorBalance(img, color_balance_clip_percentile)
    # _debug_timing("Color Balance")

    # apply CLAHE
    img2, L1, L2 = applyCLAHE(img1, clahe_clip_limit)
    # _debug_timing("ClAHE")

    # calculate normalized weight
    w1 = calcWeight(img1, L1)
    # _debug_timing("Weights 1")
    w2 = calcWeight(img2, L2)
    # _debug_timing("Weights 2")
    sum_w = w1 + w2
    w1 /= sum_w
    w2 /= sum_w
    # _debug_timing("Weights Norm")

    result = fuseTwoImage(w1, img1, w2, img2, level).clip(0, 1)
    # _debug_timing("Fuse")
    return result


# def bgr_to_lab(img: np.ndarray) -> np.ndarray:
#     img = img_to_float32(img)
#     params = np.array([
#         [0.4124530, 0.3575800, 0.1804230],
#         [0.2126710, 0.7151600, 0.0721690],
#         [0.0193340, 0.1191930, 0.950227]
#     ])
#     xyz = params @ (img.reshape(3, -2)[::-1])
#     xyz = xyz.reshape(3, *img.shape[:2])
#     xyz[0] /= 0.950456
#     xyz[2] /= 1.088754

#     result = np.zeros_like(img)
#     result[..., 0] = np.piecewise(xyz[1], [xyz[1] > 0.008856, xyz[1] <= 0.008856], [
#         lambda x: 116 * np.power(x, 1/3) - 16,
#         lambda x: 903.3 * x
#     ])
#     fxyz = np.piecewise(xyz, [xyz > 0.008856, xyz <= 0.008856], [
#         lambda x: np.power(x, 1/3),
#         lambda x: 7.787 * x + 16/116
#     ])
#     result[..., 1] = 500 * (fxyz[0] - fxyz[1])
#     result[..., 2] = 200 * (fxyz[1] - fxyz[2])
#     return result.astype(np.float32)


#########################################################
# Image utilities
#########################################################
def float_convertScaleAbs(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return np.abs(alpha * img + beta).clip(0, 255) / 255


def lab_img_to_float(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img[..., 0] /= 100
    img[..., 1:] = (img[..., 1:] + 128) / 255
    return img


def img_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    elif img.dtype in [np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        return np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype in [np.float16, np.float32, np.float64]:
        return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")


def img_to_float32(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.float16:
        return img  # TODO: clip to 0->1.0 here?
    elif img.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        return np.clip(img, 0, 255).astype(np.float32) / 255
    elif img.dtype in [np.float32, np.float64]:
        return np.clip(img, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")


#########################################################
# Fusion Enhance Helpers
#########################################################
def applyCLAHE(img, clip_limit=2.0):
    # Perform sRGB to CIE Lab color space conversion
    # _debug_timing()
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # _debug_timing("TO LAB")
    L1 = img_lab[..., 0] / 100  # convert to float 0->1 range from LAB range (0->100)
    # _debug_timing()
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    # _debug_timing("CLAHE CREATE")
    L2 = clahe.apply(img_lab[..., 0].astype(np.uint8))  # need to apply clahe to 8 bit
    # _debug_timing("CLAHE Apply")
    L2 = img_to_float32(L2)  # convert back to float
    # _debug_timing("float")
    img_lab[..., 0] = L2 * 100  # convert back to LAB range(0 -> 100)
    # _debug_timing()
    img2 = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
    # _debug_timing("From LAB")
    return img2, L1, L2


def calcWeight(img, L):  # wants float L
    # calculate laplacian contrast weight
    WL = LaplacianContrast(L)
    # calculate Local contrast weight
    WC = LocalContrast(L)
    # calculate the saliency weight
    # WS = Saliency(img)  # slower than Saturation
    WS = Saturation(img)
    # calculate the exposedness weight - no longer used in most recent paper
    # WE = Exposedness(L)
    # sum
    return WL + WC + WS  # + WE


def SimplestColorBalance(img, percent: int):
    """
    Simplest Color Balance. Performs color balancing via histogram
    normalization.

    :param img: input color or gray scale image
    :param percent: controls the percentage of pixels to clip to white and black. (normally, choose 1~10)
    :return: Balanced image in CvType.CV_32F
    """
    if percent <= 0:
        percent = 5
    halfPercent = percent / 2

    low_percentiles = np.percentile(img, halfPercent, axis=(0, 1)).astype(np.float32)
    high_percentiles = np.percentile(img, 100 - halfPercent, axis=(0, 1)).astype(np.float32)
    result = np.clip(img, low_percentiles, high_percentiles)
    return (result - low_percentiles) / (high_percentiles - low_percentiles) / 2


def fuseTwoImage_v2(w1, img1, w2, img2, level: int):
    # construct the gaussian pyramid for weight
    weight1 = buildGaussianPyramid(w1, level)
    weight2 = buildGaussianPyramid(w2, level)

    lapPyr1 = buildLaplacianPyramid(img1, level)
    lapPyr2 = buildLaplacianPyramid(img2, level)

    weightedPyr = [lapPyr1[i] * np.expand_dims(weight1[i], axis=2)
                   + lapPyr2[i] * np.expand_dims(weight2[i], axis=2) for i in range(level)]

    return reconstructLaplacianPyramid(weightedPyr)


def fuseTwoImage(w1, img1, w2, img2, level: int):
    # construct the gaussian pyramid for weight
    weight1 = buildGaussianPyramid(w1, level)
    weight2 = buildGaussianPyramid(w2, level)

    # construct the laplacian pyramid for input image channel
    bCnl1 = buildLaplacianPyramid(img1[:, :, 0], level)
    gCnl1 = buildLaplacianPyramid(img1[:, :, 1], level)
    rCnl1 = buildLaplacianPyramid(img1[:, :, 2], level)

    bCnl2 = buildLaplacianPyramid(img2[:, :, 0], level)
    gCnl2 = buildLaplacianPyramid(img2[:, :, 1], level)
    rCnl2 = buildLaplacianPyramid(img2[:, :, 2], level)
    # fusion process
    bCnl = [0] * level
    gCnl = [0] * level
    rCnl = [0] * level
    for i in range(level):
        bCnl[i] = bCnl1[i] * weight1[i] + bCnl2[i] * weight2[i]
        gCnl[i] = gCnl1[i] * weight1[i] + gCnl2[i] * weight2[i]
        rCnl[i] = rCnl1[i] * weight1[i] + rCnl2[i] * weight2[i]

    # reconstruct & output
    bChannel = reconstructLaplacianPyramid(bCnl)
    gChannel = reconstructLaplacianPyramid(gCnl)
    rChannel = reconstructLaplacianPyramid(rCnl)
    return cv2.merge((bChannel, gChannel, rChannel))


def buildGaussianPyramid(img, level: int):
    gaussPyr = [0] * level
    gaussPyr[0] = cv2.filter2D(img, -1, FILTER_MASK)
    tmpImg = img.copy()
    for i in range(1, level):
        # resize image
        tmpImg = cv2.resize(tmpImg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        # blur image
        gaussPyr[i] = cv2.filter2D(tmpImg, -1, FILTER_MASK)
    return gaussPyr


def buildLaplacianPyramid(img, level: int):
    lapPyr = [0] * level
    lapPyr[0] = img.copy()
    # resize image
    for i in range(1, level):
        lapPyr[i] = cv2.resize(lapPyr[i-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # calculate the DoG
    for i in range(level-1):
        lapPyr[i] = lapPyr[i] - cv2.resize(lapPyr[i + 1], lapPyr[i].shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return lapPyr


def reconstructLaplacianPyramid(pyramid):
    for i in range(len(pyramid)-1, 0, -1):
        pyramid[i-1] += cv2.resize(pyramid[i], pyramid[i - 1].shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return pyramid[0]


def Saliency(img: np.ndarray) -> np.ndarray:
    # blur image with a 3x3 or 5x5 Gaussian filter
    gfbgr = cv2.GaussianBlur(img, (3, 3), 3)
    # Perform sRGB to CIE Lab color space conversion
    LabIm = cv2.cvtColor(gfbgr, cv2.COLOR_BGR2Lab)
    LabIm = lab_img_to_float(LabIm)
    # Compute Lab average values (note that in the paper this average is found from the
    # un-blurred original image, but the results are quite similar)
    LabIm = LabIm - LabIm.mean(axis=(0, 1))
    # Finally compute the saliency map
    return LabIm[:, :, 0]**2 + LabIm[:, :, 1]**2 + LabIm[:, :, 2]**2


def Saturation(img: np.ndarray) -> np.ndarray:
    result = img - img.mean(axis=(0, 1))
    return np.sqrt(result[..., 0]**2 + result[..., 1]**2 + result[..., 2]**2)


def LaplacianContrast(img: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(img, -1)
    #Imgproc.Laplacian(img, laplacian, img.depth(), 3, 1, 0);
    # return cv2.convertScaleAbs(laplacian)
    return float_convertScaleAbs(laplacian)


def LocalContrast(img: np.ndarray) -> np.ndarray:
    localContrast = cv2.filter2D(img, -1, FILTER_MASK)
    localContrast = np.clip(localContrast, None, math.pi / 2.75)
    localContrast = img - localContrast
    return localContrast ** 2


def Exposedness(img: np.ndarray, average=0.5, sigma=0.25) -> np.ndarray:
    # W = exp(-(img - aver).^2 / (2*sigma^2));
    return np.exp(-1.0 * ((img - average) ** 2) / (2 * (sigma**2)))
