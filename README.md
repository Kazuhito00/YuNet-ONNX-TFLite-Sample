# YuNet-ONNX-TFLite-Sample
[YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)のPythonでのONNX、TensorFlow-Lite推論サンプルです。<br>
TensorFlow-Liteモデルは[PINTO0309/PINTO_model_zoo/144_YuNet](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/144_YuNet)のものを使用しています。<br>

<img src="https://user-images.githubusercontent.com/37477845/136249513-1aed003a-dc3b-4e8e-9143-3f134b252969.gif" width="50%">

# Requirement 
* OpenCV 3.4.2 or later
* onnxruntime 1.5.2 or later
* tensorflow 2.6.0 or later

# Demo
デモの実行方法は以下です。
```bash
python sample_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/face_detection_yunet_120x160.onnx
* --input_shape<br>
モデルの入力サイズ<br>
デフォルト：160,120
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.6
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.3
* --topk<br>
topk指定値<br>
デフォルト：5000
* --keep_topk<br>
keep_topk指定値<br>
デフォルト：750

```bash
python sample_tlite.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/model_float16_quant.tflite
* --input_shape<br>
モデルの入力サイズ<br>
デフォルト：160,120
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.6
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.3
* --topk<br>
topk指定値<br>
デフォルト：5000
* --keep_topk<br>
keep_topk指定値<br>
デフォルト：750

# Reference
* [opencv/opencv_zoo/YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
* [PINTO0309/PINTO_model_zoo/144_YuNet](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/144_YuNet)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
YuNet-ONNX-TFLite-Sample is under [Apache-2.0 License](LICENSE).
