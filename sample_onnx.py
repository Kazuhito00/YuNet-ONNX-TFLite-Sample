#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv

from yunet.yunet_onnx import YuNetONNX


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/face_detection_yunet_120x160.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="160,120",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.6,
        help='Conf confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.3,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--keep_topk',
        type=int,
        default=750,
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    topk = args.topk
    keep_topk = args.keep_topk

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    yunet = YuNetONNX(
        model_path=model_path,
        input_shape=input_shape,
        conf_th=score_th,
        nms_th=nms_th,
        topk=topk,
        keep_topk=keep_topk,
    )

    while True:
        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        bboxes, landmarks, scores = yunet.inference(frame)

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            input_shape,
            bboxes,
            landmarks,
            scores,
        )

        # キー処理(ESC：終了) ##############################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #########################################################
        cv.imshow('YuNet ONNX Sample', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_debug(
    image,
    elapsed_time,
    score_th,
    input_shape,
    bboxes,
    landmarks,
    scores,
):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    for bbox, landmark, score in zip(bboxes, landmarks, scores):
        if score_th > score:
            continue

        # 顔バウンディングボックス
        x1 = int(image_width * (bbox[0] / input_shape[0]))
        y1 = int(image_height * (bbox[1] / input_shape[1]))
        x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
        y2 = int(image_height * (bbox[3] / input_shape[1])) + y1

        cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # スコア
        cv.putText(debug_image, '{:.4f}'.format(score), (x1, y1 + 12),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        # 顔キーポイント
        for _, landmark_point in enumerate(landmark):
            x = int(image_width * (landmark_point[0] / input_shape[0]))
            y = int(image_height * (landmark_point[1] / input_shape[1]))
            cv.circle(debug_image, (x, y), 2, (0, 255, 0), 2)

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv.putText(
        debug_image,
        text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
