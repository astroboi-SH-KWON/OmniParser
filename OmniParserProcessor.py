from typing import Optional
import io
import base64
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image
import cv2

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

DEVICE = torch.device('cuda')


# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(image_input, box_threshold=0.05, iou_threshold=0.1, use_paddleocr=True, imgsz=640) -> Optional[Image.Image]:
    image_save_path = 'imgs/saved_image_demo.png'
    cv2.imwrite(image_save_path, image_input)
    # image_input.save(image_save_path)  # AttributeError: 'numpy.ndarray' object has no attribute 'save'
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 4 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    # # OCR
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img=False, output_bb_format='xyxy',
                                                    goal_filtering=None,
                                                    easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                                                    use_paddleocr=use_paddleocr)
    # # ocr_result, bbox_coordinates
    text, ocr_bbox = ocr_bbox_rslt
    # print('prompt:', prompt)
    # # parsed_content_list : contents explanation (Image Captioning)
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model,
                                                                                  BOX_TRESHOLD=box_threshold,
                                                                                  output_coord_in_ratio=True,
                                                                                  ocr_bbox=ocr_bbox,
                                                                                  draw_bbox_config=draw_bbox_config,
                                                                                  caption_model_processor=caption_model_processor,
                                                                                  ocr_text=text,
                                                                                  iou_threshold=iou_threshold,
                                                                                  imgsz=imgsz, )
    print('finish processing')
    parsed_content_list = [
        {'type': v_dict['type'], 'interactivity': v_dict['interactivity'], 'content': v_dict['content']} for v_dict in
        parsed_content_list if isinstance(v_dict, dict)]
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    parsed_content_dict ={}
    for i, v in enumerate(parsed_content_list):
        parsed_content_dict.update({f'icon {i}': v})

    return dino_labled_img, str(parsed_content_list)
