from util import utils
from flask import Flask, request, render_template, abort, make_response
import time
import json
import base64
import uuid
import os
import cv2
import platform
from OmniParserProcessor import process


app = Flask('OmniParser')
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=['GET', 'POST'])
def omniparser_api():
    st_time = time.perf_counter()
    data = request.data.decode('utf-8')
    data = json.loads(data)
    img_data = data['screenshot']
    box_threshold = data['box_threshold']
    iou_threshold = data['iou_threshold']
    which_ocr = data['which_ocr']
    imgsz = data['img_size']

    use_paddleocr = False
    if which_ocr == "paddleocr":
        use_paddleocr = True

    if utils.is_hexadecimal(img_data):
        print("is_hexadecimal >>>>>>>>>>>>>>>>>>>>>>>>")
        img_bytes = bytes.fromhex(img_data)
        img_obj = utils.bts_to_img(img_bytes)
    elif utils.is_base64(img_data):
        print("is_Base64 >>>>>>>>>>>>>>>>>>>>>>>>")
        img_bytes = base64.b64decode(img_data)
        img_obj = utils.bts_to_img(img_bytes)
    else:
        print("[ERROR-feature_extractor_api] Choose decoding options")
        raise Exception

    try:
        fl_nm = str(uuid.uuid4())
        os.makedirs("images/tmp", exist_ok=True)
        tmp_fl = f"images/tmp/img_{fl_nm}.jpg"
        cv2.imwrite(tmp_fl, img_obj)
        loaded_img = utils.load_img_by_cv2(tmp_fl)
        if os.path.isfile(tmp_fl):
            os.remove(tmp_fl)
    except Exception as err:
        print(err)
        return abort(make_response(str(err), 500))

    ef_st_time = time.perf_counter()
    img, text = process(loaded_img, box_threshold, iou_threshold, use_paddleocr, imgsz)
    result = {
        "res_img": img,
        "res_text": text,
    }
    # print(f"omniparser ::: {time.perf_counter() - ef_st_time} sec\n{result}\n\n")
    print(f"omniparser ::: {time.perf_counter() - ef_st_time}")
    return {'result': json.dumps(result)}


if __name__ == '__main__':
    port = 8025
    os_name = platform.system()
    # app.debug = True
    if os_name == "Linux":
        app.run(debug=True, host='192.168.3.138', port=port)
    elif os_name == "Darwin" or os_name == "Windows":
        app.run(debug=True, host='127.0.0.1', port=port)
    else:
        app.run(debug=True, host='0.0.0.0', port=port)
