import argparse
import pandas as pd
import cv2
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--normalize_predictions', action="store_true", default=False)
    parser.add_argument('--dataset_folder', type=str, default=None, help="Required if --normalize_predictions is set")
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = get_args()

    assert args.normalize_predictions and (args.dataset_folder is not None), "If --normalize_predictions is set, --dataset_folder must be set"
    predictions = pd.read_csv(args.input, names=["image_id", "class", "confidence", "x", "y", "w", "h"])
    
    if args.normalize_predictions:
        predictions["im_shape"] = predictions["image_id"].apply(lambda im_id: cv2.imread(os.path.join(args.dataset_folder, str(im_id) + ".jpg")).shape[:2])
        predictions["x"] = predictions["x"] / predictions["im_shape"].apply(lambda x: x[1])
        predictions["y"] = predictions["y"] / predictions["im_shape"].apply(lambda x: x[0])
        predictions["w"] = predictions["w"] / predictions["im_shape"].apply(lambda x: x[1])
        predictions["h"] = predictions["h"] / predictions["im_shape"].apply(lambda x: x[0])
    
    predictions.to_csv(args.output, index=False, header=True)
    