import logging
import cv2
import numpy as np
from torchvision import transforms

logger = logging.getLogger(__name__)

def unnormalize(img_base):
    aug_mean = np.array([0.485, 0.456, 0.406])
    aug_std = np.array([0.229, 0.224, 0.225])
    unnormalize = transforms.Normalize((-aug_mean / aug_std).tolist(), (1.0 / aug_std).tolist())
    img_unnorm = unnormalize(img_base)

    return img_unnorm

def resize_image(image, new_height):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

## Functions for handling rotated bounding boxes

def rotate_box(x1,y1,x2,y2,theta):
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2

    h = int(y2 - y1)
    w = int(x2 - x1)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    A = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    C = np.array([[xm, ym]])
    RA = (A - C) @ R.T + C
    RA = RA.astype(int)

    return RA

def crop_rect(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]
    
    diag_len = int(np.sqrt(height * height + width * width))
    new_width = diag_len
    new_height = diag_len

    blank_canvas = np.ones((new_height, new_width, 3), dtype=img.dtype) * 255

    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2

    blank_canvas[y_offset:y_offset+height, x_offset:x_offset+width] = img

    new_center_x = new_width // 2
    new_center_y = new_height // 2

    M = cv2.getRotationMatrix2D((new_center_x, new_center_y), np.rad2deg(angle), 1)

    img_rot = cv2.warpAffine(blank_canvas, M, (new_width, new_height), flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

    new_center = np.dot(M[:,:2], np.array([center[0], center[1]]) + np.array([x_offset, y_offset])) + M[:,2]

    img_crop = cv2.getRectSubPix(img_rot, size, new_center)
    return img_crop, img_rot


def get_chip_from_img(img, bbox, theta):
    x1,y1,w,h = bbox

    # Degenerate zero-size bbox falls back to the original image. The
    # theta=0 path below already handles this implicitly via the empty
    # numpy slice + min(shape) check, but the rotated path would hand
    # cv2.getRectSubPix a size of (0, 0), which returns None and then
    # crashes the .shape check. Bail out before either path runs.
    if w <= 0 or h <= 0:
        logger.warning(f'Using original image. Zero-size bbox: {bbox}')
        return img

    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2

    # Fast path only for a box that is genuinely axis-aligned AND wholly inside
    # the frame. The old `abs(theta) < 0.1` tolerance silently discarded up to
    # 5.7 degrees of rotation -- enough to clip a long, narrow animal -- and
    # clamping a negative origin with max(0, ...) while keeping the full width
    # SHIFTS the window onto a different region rather than padding it. An
    # object-aligned box legitimately overhangs the frame, so both cases now go
    # through crop_rect, which samples the requested rectangle off a padded
    # canvas. Detector boxes carry theta of exactly 0.0 and are already clamped
    # in-frame, so they keep the cheap slice.
    img_h, img_w = img.shape[0], img.shape[1]
    wholly_inside = (x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h)
    if theta == 0.0 and wholly_inside:
        xi, yi, wi, hi = [int(v) for v in bbox]
        cropped_image = img[yi : yi + hi, xi : xi + wi]
    elif theta == 0.0:
        # Zero-angle overhang. crop_rect would produce the right pixels, but it
        # builds a diagonal-sized canvas AND warps the whole image to do it --
        # about 156MB per buffer on a 6000x4000 RGB frame. YOLO dilation is not
        # clamped to the frame, so existing callers really do land here.
        #
        # We keep cv2.getRectSubPix and only shrink what it reads from. A plain
        # slice is NOT equivalent: getRectSubPix samples from
        # center - (size-1)/2, so an integer centre with an EVEN dimension
        # lands on a half-pixel and interpolates against the white border.
        # Sizes and centre are derived exactly as crop_rect derives them.
        iw, ih = int(x2 - x1), int(y2 - y1)
        cx, cy = int(xm), int(ym)
        # The window getRectSubPix will read, plus one pixel for the
        # interpolation tap on each side.
        sx = cx - (iw - 1) / 2.0
        sy = cy - (ih - 1) / 2.0
        px0, py0 = int(np.floor(sx)) - 1, int(np.floor(sy)) - 1
        px1, py1 = int(np.ceil(sx + iw)) + 1, int(np.ceil(sy + ih)) + 1
        # `img.shape[2:]` not `img.shape[2]`: a plain (H, W) grayscale frame has
        # no channel axis and would raise.
        patch = np.full((py1 - py0, px1 - px0) + img.shape[2:], 255, dtype=img.dtype)
        ox1, oy1 = max(0, px0), max(0, py0)
        ox2, oy2 = min(img_w, px1), min(img_h, py1)
        if ox2 > ox1 and oy2 > oy1:
            patch[oy1 - py0:oy2 - py0, ox1 - px0:ox2 - px0] = img[oy1:oy2, ox1:ox2]
        cropped_image = cv2.getRectSubPix(patch, (iw, ih), (cx - px0, cy - py0))
    else:
        cropped_image = crop_rect(img, ((xm, ym), (x2-x1, y2-y1), theta))[0]

    if cropped_image is None or min(cropped_image.shape) < 1:
        # Use original image
        logger.warning(f'Using original image. Invalid parameters - theta: {theta}, bbox: {bbox}')
        cropped_image = img

    return cropped_image

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image