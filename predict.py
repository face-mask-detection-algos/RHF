import os
import time
import json

import argparse
import cv2
import numpy as np

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box

def get_args():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument("--weigths_path", type=str, default=r"./save_weights/resNetFpn-model-19.pth", help="path to weights file (default ./save_weights/resNetFpn-model-19.pth)")
    parser.add_argument("--from_webcam", action="store_true", default=False, help="use video source from webcam (default False). It overrides possible images indicated in --images_folder")
    parser.add_argument("--camera_id", type=int, default=0, help="if --from_webcam active, specifies camera id (default 0)")
    parser.add_argument("--images_folder", type=str, default=r"./test_images", help="path to images (whole folder - default ./test_images)")
    parser.add_argument("--output_folder", type=str, default=None, help="folder where images will be saved (default None). If not saved, images will be shown in a window")
    args = parser.parse_args()
    return args

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    args = get_args()
    if args.output_folder is not None:
        os.makedirs(args.output_folder, exist_ok=True)
    
    
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=4)

    # load train weights
    train_weights = args.weigths_path
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    if args.from_webcam:
        img_source = cv2.VideoCapture(args.camera_id)
    else:
        img_source = [os.path.join(args.images_folder, img) for img in sorted(os.listdir(args.images_folder))]
    img_id = 0
    
    # load image
    # original_img = Image.open("test_images/test1.jpg")
    
    while True:
        if args.from_webcam:
            ret, original_img = img_source.read()
            if not ret:
                raise IOError("Couldn't open webcam or video")
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_img = Image.fromarray(original_img)
            img_name = f"webcam_{img_id}.jpg"
        else:
            try:
                original_img = Image.open(img_source[img_id])
                img_name = os.path.splitext(os.path.basename(img_source[img_id]))[0]
                print(f"predicting image {img_name}")
            except IndexError:
                break
            
            

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(original_img,
                    predict_boxes,
                    predict_classes,
                    predict_scores,
                    category_index,
                    thresh=0.5,
                    line_thickness=3)
            # plt.imshow(original_img)
            # plt.show()
            # # 保存预测的图片结果
            # original_img.save("test_images/test1_result_ResNet50_NotAugment.jpg")
        
        if args.output_folder is None:
            cv2.imshow("result", cv2.cvtColor(np.asarray(original_img), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            original_img.save(os.path.join(args.output_folder, img_name + ".jpg"))    
        img_id += 1
        


if __name__ == '__main__':
    main()

