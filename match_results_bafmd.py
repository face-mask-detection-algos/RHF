import pandas as pd
import argparse
import numpy as np
from scipy import stats
import os

def print_results(dict_of_dfs):
    for attr, df in dict_of_dfs.items():
        print("Attribute: {}".format(attr))
        print(df)
        print("")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--ground_truth', type=str, required=True)
    parser.add_argument('--output', type=str, default="fairness_metrics.csv")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--additional_localization_analysis", action="store_true", default=False, help="If set, additional analysis on localization will be performed")
    return parser.parse_args()

def compute_iou(x1, y1, h1, w1, x2, y2, h2, w2):
    ''' Given the coordinates of the bounding boxes, compute the IoU '''
    xb1 = max(x1, x2)
    yb1 = max(y1, y2)
    xb2 = min(x1 + w1, x2 + w2)
    yb2 = min(y1 + h1, y2 + h2)
    
    intersection = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    return intersection / (area1 + area2 - intersection)


def get_stats(ground_truth, subset, attribute_list, invert_prop=False, ):
    '''
    Get fairness on attribute_list.
    
    attribute_list can be a list or a dict. If list, the attribute has the
    same name on both ground_truth and subset. If dict, specify as keys the
    name on ground_truth, and as values the name on subset.
    '''
    n_tot = len(ground_truth)
    n_subset = len(subset)
    
    if isinstance(attribute_list, list):
        attribute_list = {attribute: attribute for attribute in attribute_list}
    
    stats_per_attribute = {}
    
    for attribute_gt, attribute_sub in attribute_list.items():
        attr_count_gt = ground_truth.groupby(attribute_gt).count().iloc[:, 0]
        attr_count_sub = subset.groupby(attribute_sub).count().iloc[:, 0]
        df = pd.DataFrame({"ground_truth": attr_count_gt, "subset": attr_count_sub}).fillna(0)
        df["prop"] = df["subset"] / df["ground_truth"]
        df["s_other"] = n_subset - df["subset"]
        df["n_other"] = n_tot - df["ground_truth"]
        df["p_other"] = df["s_other"] / df["n_other"]
        if invert_prop:
            df["prop"] = 1 - df["prop"]
            df["p_other"] = 1 - df["p_other"]
        df["pooled_p"] = (df.ground_truth * df.prop + df.n_other * df.p_other)/(df.ground_truth + df.n_other)
        df["test_stat"] = (df.prop - df.p_other) / np.sqrt(df.pooled_p * (1 - df.pooled_p) * (1/df.ground_truth + 1/df.n_other))
        df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(df.test_stat)))
        stats_per_attribute[attribute_gt] = df
    
    return stats_per_attribute
        

if __name__ == "__main__":
    args = get_args()
    ground_truth = pd.read_csv(args.ground_truth)
    ground_truth = ground_truth.drop(ground_truth.columns[0], axis=1)
    ground_truth["filename"] = ground_truth["filename"].apply(lambda x: x.replace(".jpg", ""))
    # rewrite ground truth to match predictions
    ground_truth["x"] = ground_truth["x"] - ground_truth["dx"]/2
    ground_truth["y"] = ground_truth["y"] - ground_truth["dy"]/2
       
    predictions = pd.read_csv(args.predictions).drop("im_shape", axis=1)
    predictions = predictions.rename(columns={"image_id": "filename"})
    predictions["class"] = predictions["class"].apply(lambda x: 1 if x in (1, 2) else 0)
    # adjust predictions
    predictions["h"] = predictions["h"] - predictions["y"]
    predictions["w"] = predictions["w"] - predictions["x"]
    
    # couple each ground truth bounding box with each prediction bounding box for the same image
    join = pd.merge(ground_truth, predictions, on="filename", how="left")
    join["iou"] = join.apply(lambda x: compute_iou(x["x_x"], x["y_x"], x["dy"], x["dx"], x["x_y"], x["y_y"], x["h"], x["w"]), axis=1)
    # filter out those with IoU > threshold
    join_matches = join[join["iou"] > args.iou_threshold]
    
    # get missed localizations
    non_localized = ground_truth.merge(join_matches, how="left", indicator=True, right_on=["filename", "x_x", "y_x", "dx", "dy"], left_on=["filename", "x", "y", "dx", "dy"]).query("_merge == 'left_only'").drop("_merge", axis=1)
    
    
    # keep only the best match for each ground truth bounding box
    join_matches = join_matches.sort_values("iou", ascending=False).drop_duplicates(subset=["filename", "x_x", "y_x", "dx", "dy"])
    
    # filter out predictions with same bounding box but different class (break tie by keeping the one with higher confidence)
    join_matches = join_matches.sort_values("confidence", ascending=False).drop_duplicates(subset=["filename", "x_x", "y_x", "dx", "dy", "x_y", "y_y", "w", "h"])
    
    # get true positives and true negatives
    true_positives = join_matches.rename(columns={"class": "class_pred"}).query("mask == 1 and class_pred == 1")
    gt_positives = join_matches.rename(columns={"class": "class_pred"}).query("mask == 1")
    
    true_negatives = join_matches.rename(columns={"class": "class_pred"}).query("mask == 0 and class_pred == 0")
    gt_negatives = join_matches.rename(columns={"class": "class_pred"}).query("mask == 0")
    
    # get stats
    df_localization = get_stats(ground_truth, non_localized, invert_prop=True, attribute_list={"skincolor": "skincolor_x", "sex": "sex_x"})
    df_positives = get_stats(gt_positives, true_positives, attribute_list=["skincolor", "sex"])
    df_negatives = get_stats(gt_negatives, true_negatives, attribute_list=["skincolor", "sex"])
    
    if args.additional_localization_analysis:
        df_mask_loc = get_stats(ground_truth, join_matches, attribute_list=["mask"])
    
    print("Localization", "\n", "------------------")
    print_results(df_localization)
    print("Positives", "\n", "------------------")
    print_results(df_positives)
    print("Negatives", "\n", "------------------")
    print_results(df_negatives)
    
    if args.additional_localization_analysis:
        print("Mask localization", "\n", "------------------")
        print_results(df_mask_loc)
    
    df_analysis = [df_localization, df_positives, df_negatives]
    name_analysis = ["localization", "positives", "negatives"]
    if args.additional_localization_analysis:
        df_analysis += [df_mask_loc]
        name_analysis += ["mask_localization"]
        
    
    # save results
    for df, variable in zip(df_analysis, name_analysis):
        for attribute, stats in df.items():
            output_dir = od if (od:=os.path.dirname(args.output)) != "" else "."
            output_name, output_extension = os.path.splitext(os.path.basename(args.output))
            stats.to_csv(f"{output_dir}/{output_name}_{attribute}_{variable}.csv")
    
    
    
    
    