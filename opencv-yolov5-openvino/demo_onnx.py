import time

import cv2
import onnx
import onnxruntime
import torch
from argparse import ArgumentParser
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

def build_argparser():
    parser = ArgumentParser(prog="demo_onnx.py")
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/solarcell.onnx",
        help="model.xml path",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=r"C:\Users\zengxh\Documents\workspace\visualstudio-workspace\libtorch-demo\opencv-yolov5-openvino\images\quejiao.jpg",
        help="image path",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.1, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.1, help="IOU threshold for NMS"
    )
    parser.add_argument("--cv", action="store_true", help="use PIL or opencv image")

    return parser

def resize(image, size, interpolation=cv2.INTER_CUBIC):
    shape = image.shape[0:2][::-1]
    iw, ih = shape
    w, h = size
    scale = min(w / iw, h / ih)
    ratio = scale, scale
    nw = int(round(iw * scale))
    nh = int(round(ih * scale))
    new_unpad = nw, nh

    if shape[0:2] != new_unpad:
        image = cv2.resize(image, (nw, nh), interpolation=interpolation)

    new_image = np.full((size[1], size[0], 3), 128)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy : dy + nh, dx : dx + nw, :] = image

    return new_image, ratio, (dx, dy)

def letterbox_image(pil_img, size):
    iw, ih = pil_img.size
    w, h = size
    scale = min(w / iw, h / ih)
    ratio = scale, scale
    nw = int(iw * scale)
    nh = int(ih * scale)
    dw = (w - nw) // 2
    dh = (h - nh) // 2

    pil_img = pil_img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(pil_img, (dw, dh))
    return new_image, ratio, (dw, dh)

def preprocess_cv_img(image_src, input_height, input_width, nchw_shape=True):
    in_image_src, _, _ = resize(image_src, (input_width, input_height))
    in_image_src = in_image_src[..., ::-1]
    if nchw_shape:
        in_image_src = in_image_src.transpose(
            (2, 0, 1)
        )  # Change data layout from HWC to CHW
    # in_image_src = np.ascontiguousarray(in_image_src)
    in_image_src = np.expand_dims(in_image_src, axis=0)

    img_in = in_image_src / 255.0
    return img_in


def preprocess_pil_img(image_src, input_size):
    resized, _, _ = letterbox_image(image_src, (input_size, input_size))

    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    img_in = np.expand_dims(img_in, axis=0)

    return img_in

def main(args):
    session = onnxruntime.InferenceSession(args.model_path)

    if args.cv:
        image_src = cv2.imread(args.image_path).astype(np.float32)
        img_in = preprocess_cv_img(image_src, args.img_size, args.img_size)
    else:
        image_src = Image.open(args.image_path)
        img_in = preprocess_pil_img(image_src, args.img_size)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    detections = non_max_suppression(
        outputs[0], conf_thres=args.conf_thres, iou_thres=args.iou_thres, agnostic=False
    )

    return detections

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(
    prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False
):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        if type(x).__module__ == "numpy":
            x = torch.from_numpy(x)
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)

def display(
    detections=None,
    image_path=None,
    input_size=640,
    line_thickness=None,
    text_bg_alpha=0.0,
):
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    image_src = Image.open(image_path)
    w, h = image_src.size
    image_src = np.array(image_src)

    boxs[:, :] = scale_coords((input_size, input_size), boxs[:, :], (h, w)).round()

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = box
        np.random.seed(int(labels[i].numpy()) + 2020)
        color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
        cv2.rectangle(
            image_src,
            (x1, y1),
            (x2, y2),
            color,
            thickness=max(int((w + h) / 600), 1),
            lineType=cv2.LINE_AA,
        )
        label = "%s %.2f" % (class_names[int(labels[i].numpy())], confs[i])
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        if text_bg_alpha == 0.0:
            cv2.rectangle(image_src, (x1 - 1, y1), c2, color, cv2.FILLED, cv2.LINE_AA)
        else:
            alphaReserve = text_bg_alpha
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
            xMax, yMax = int(x1 + t_size[0]), int(y1)
            image_src[yMin:yMax, xMin:xMax, 0] = image_src[
                yMin:yMax, xMin:xMax, 0
            ] * alphaReserve + BChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 1] = image_src[
                yMin:yMax, xMin:xMax, 1
            ] * alphaReserve + GChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 2] = image_src[
                yMin:yMax, xMin:xMax, 2
            ] * alphaReserve + RChannel * (1 - alphaReserve)
        cv2.putText(
            image_src,
            label,
            (x1 + 3, y1 - 4),
            0,
            tl / 3,
            [255, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        print(box.numpy(), confs[i].numpy(), class_names[int(labels[i].numpy())])

    plt.imshow(image_src)
    plt.show()

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

if __name__ == "__main__":
    args = build_argparser().parse_args()
    class_names = [
        "lubai", "yiwu", "quejiao",
        "liepian", "zangwu"
    ]
    with torch.no_grad():
        detections = main(args)
        if detections[0] is not None:
            display(
                detections[0],
                args.image_path,
                input_size=args.img_size,
                text_bg_alpha=0.6,
            )