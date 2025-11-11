import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os, sys
from PIL import Image
from torchvision.transforms import ToPILImage

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import longclip

device = "cuda:5"
clip_model, clip_processor = longclip.load("../longclip/longclip-L.pt", device=device)

model = YOLO('yolov8x-seg.pt')


def instance_segment(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    segmented_images = []

    if results[0].masks == None:
        return segmented_images

    for idx, result in enumerate(results):
        masks = result.masks.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        if boxes[0][2] - boxes[0][0] < 50 or boxes[0][3] - boxes[0][1] < 50:
            return segmented_images
        for mask_idx, mask, box in zip(range(len(masks)), masks, boxes):
            try:
                mask = cv2.resize(mask.data.astype(np.uint8)[0], (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            except:
                return segmented_images
            mask = mask.astype(bool)
            segmented_image = np.zeros_like(image)
            segmented_image[mask] = image[mask]

            segmented_image2 = np.transpose(segmented_image, (2, 0, 1))

            segmented_images.append(segmented_image2)

    #         output_path = f'segmented_output_{idx + 1}_{mask_idx + 1}.jpg'
    #         cv2.imwrite(output_path, segmented_image)
    #         print(f'Saved: {output_path}')
    # print('All cropped instances saved successfully.')
    return segmented_images


def encode_image(image):
    # image2 = './1.jpg'
    # image3 = Image.open(image2)
    to_pil_image = ToPILImage()
    tensor_image = torch.tensor(image[0])
    pil_image = to_pil_image(tensor_image)
    img_processed = clip_processor(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = clip_model.encode_image(img_processed)
    return image_feature


def encode_text(text):
    inputs = longclip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(inputs).to(device)
    return text_features


def segment(query, image, output_path=None):
    seg_imgs = instance_segment(image)
    query_vec = encode_text(query)
    cross_result = []
    for seg_img in seg_imgs:
        img_vec = encode_image(seg_imgs)
        cross_sim = torch.cosine_similarity(query_vec, img_vec).item()
        if cross_sim >= 0.19:
            cross_result.append(seg_img)
    if len(cross_result) == 1:
        # seg_list.append(image)
        image_name = image.split('/')[-1]
        if output_path == None:
            output_path = f'./image_seg/{image_name}'

        segmented_image = np.transpose(cross_result[0], (1, 2, 0))
        cv2.imwrite(output_path, segmented_image)
        return True
    else:
        return False


if __name__ == '__main__':
    image = './3.jpg'
    query = 'How many teeth does this animal use to have?'
    segment(query, image, output_path='3_seg.jpg')

# Exclude: no segmentation results, segmentation results with too low similarity to query, entities outside top 3
