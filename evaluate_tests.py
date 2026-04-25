"""
Script to evaluate the submission against the ground truth annotations for the AV2 motion forecasting challenge.
"""

from av2.datasets.motion_forecasting.eval import metrics

import pandas as pd
import numpy as np 

from tqdm import tqdm


test_anno_path = "/workspaces/datasets/av2_mf_focal_test_annotations.parquet"
submission_path = "/workspaces/QCNet/submission.parquet"

gt_df = pd.read_parquet(test_anno_path)
submission_df = pd.read_parquet(submission_path)

gt_df_grouped =gt_df.groupby(["scenario_id", "track_id"])
submission_df_grouped = submission_df.groupby(["scenario_id", "track_id"])

metrics_df = pd.DataFrame(columns=["scenario_id", "track_id", "minADE", "minFDE", "brier_minADE", "brier_minFDE", "is_missed_prediction"])

a = True

for id_tuple, group in tqdm(gt_df_grouped):
    (scenario_id, track_id) = id_tuple

    pred = submission_df_grouped.get_group(id_tuple)
    gt = group # gt_df_grouped.get_group(id_tuple)
    
    pred_track = np.moveaxis(
        np.array(
            [[* pred["predicted_trajectory_x"]], 
            [*pred["predicted_trajectory_y"]]]
        ),
        source=[0, 1, 2], 
        destination=[2, 0, 1],
    )
    pred_prob = pred["probability"].to_numpy()

    gt_track = np.array([gt["gt_trajectory_x"].iloc[0], gt["gt_trajectory_y"].iloc[0]]).T

    if a:
        print(pred_prob)
        print([(pr, pt[-1]) for (pr, pt) in zip(pred["probability"], pred["predicted_trajectory_x"])])
        print([pred_track[:, 59, 0]])
        a = False

    ade = metrics.compute_ade(pred_track, gt_track)
    fde = metrics.compute_fde(pred_track, gt_track)
    
    metrics_df_row = {
        "scenario_id": scenario_id,
        "track_id": track_id,
        "minADE": ade.min(),
        "minFDE": fde.min(),
        "brier_minADE": ade[ade.argmin()] + (1.0 - pred_prob[ade.argmin()])**2,
        "brier_minFDE": fde[fde.argmin()] + (1.0 - pred_prob[fde.argmin()])**2,
        "is_missed_prediction": metrics.compute_is_missed_prediction(pred_track, gt_track).min(),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_df_row, index=[0])], ignore_index=True)

print(metrics_df[["minADE", "brier_minFDE", "minFDE", "brier_minADE", "is_missed_prediction"]].mean())

