import argparse
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_log_dir", type=str, help="Path to the directory containing the model checkpoints"
)


def compare_id_ood_scores(
    train_random_length_analysis_list,
    test_random_length_analysis_list,
    test_historical_length_analysis_list,
    test_inductive_length_analysis_list,
):
    """Print out the average precision scores under ID/OOD (based on temporal context length)
    settings for source and destination nodes in the test sets."""

    num_pos_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    num_random_neg_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    num_historical_neg_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    num_inductive_neg_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}

    # avg_score_pos = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    # avg_score_random_neg = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    # avg_score_historical_neg = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    # avg_score_inductive_neg = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}

    random_ap_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    historical_ap_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}
    inductive_ap_dict = {"ood_ood": [], "id_ood": [], "ood_id": [], "id_id": []}

    for (
        train_random_length_analysis,
        test_random_length_analysis,
        test_historical_length_analysis,
        test_inductive_length_analysis,
    ) in zip(
        train_random_length_analysis_list,
        test_random_length_analysis_list,
        test_historical_length_analysis_list,
        test_inductive_length_analysis_list,
    ):
        # Find the largest seen time diffs in the training sets for both source and destination nodes
        train_src_max_time_diff, train_dst_max_time_diff = np.nanmax(
            train_random_length_analysis["pos"]["src"]["max_time_diffs"]
        ), np.nanmax(train_random_length_analysis["pos"]["dst"]["max_time_diffs"])

        # Create a mask for test samples that contain time diffs larger than the largest seen time diffs in the training sets, call them OOD samples
        test_pos_src_ood_mask = (
            test_random_length_analysis["pos"]["src"]["max_time_diffs"] > train_src_max_time_diff
        )
        test_pos_dst_ood_mask = (
            test_random_length_analysis["pos"]["dst"]["max_time_diffs"] > train_dst_max_time_diff
        )

        test_random_neg_src_ood_mask = (
            test_random_length_analysis["neg"]["src"]["max_time_diffs"] > train_src_max_time_diff
        )
        test_random_neg_dst_ood_mask = (
            test_random_length_analysis["neg"]["dst"]["max_time_diffs"] > train_dst_max_time_diff
        )

        test_historical_neg_src_ood_mask = (
            test_historical_length_analysis["neg"]["src"]["max_time_diffs"]
            > train_src_max_time_diff
        )
        test_historical_neg_dst_ood_mask = (
            test_historical_length_analysis["neg"]["dst"]["max_time_diffs"]
            > train_dst_max_time_diff
        )

        test_inductive_neg_src_ood_mask = (
            test_inductive_length_analysis["neg"]["src"]["max_time_diffs"]
            > train_src_max_time_diff
        )
        test_inductive_neg_dst_ood_mask = (
            test_inductive_length_analysis["neg"]["dst"]["max_time_diffs"]
            > train_dst_max_time_diff
        )

        # Calculate the number of (OOD, OOD), (ID, OOD), (OOD, ID), (ID, ID) samples in the test positive edges, test random negative edges, test historical negative edges, and test inductive negative edges

        num_pos_ood_ood = np.logical_and(test_pos_src_ood_mask, test_pos_dst_ood_mask).sum()
        num_pos_id_ood = np.logical_and(~test_pos_src_ood_mask, test_pos_dst_ood_mask).sum()
        num_pos_ood_id = np.logical_and(test_pos_src_ood_mask, ~test_pos_dst_ood_mask).sum()
        num_pos_id_id = np.logical_and(~test_pos_src_ood_mask, ~test_pos_dst_ood_mask).sum()
        num_pos_dict["ood_ood"].append(num_pos_ood_ood)
        num_pos_dict["id_ood"].append(num_pos_id_ood)
        num_pos_dict["ood_id"].append(num_pos_ood_id)
        num_pos_dict["id_id"].append(num_pos_id_id)

        num_random_neg_ood_ood = np.logical_and(
            test_random_neg_src_ood_mask, test_random_neg_dst_ood_mask
        ).sum()
        num_random_neg_id_ood = np.logical_and(
            ~test_random_neg_src_ood_mask, test_random_neg_dst_ood_mask
        ).sum()
        num_random_neg_ood_id = np.logical_and(
            test_random_neg_src_ood_mask, ~test_random_neg_dst_ood_mask
        ).sum()
        num_random_neg_id_id = np.logical_and(
            ~test_random_neg_src_ood_mask, ~test_random_neg_dst_ood_mask
        ).sum()
        num_random_neg_dict["ood_ood"].append(num_random_neg_ood_ood)
        num_random_neg_dict["id_ood"].append(num_random_neg_id_ood)
        num_random_neg_dict["ood_id"].append(num_random_neg_ood_id)
        num_random_neg_dict["id_id"].append(num_random_neg_id_id)

        num_historical_neg_ood_ood = np.logical_and(
            test_historical_neg_src_ood_mask, test_historical_neg_dst_ood_mask
        ).sum()
        num_historical_neg_id_ood = np.logical_and(
            ~test_historical_neg_src_ood_mask, test_historical_neg_dst_ood_mask
        ).sum()
        num_historical_neg_ood_id = np.logical_and(
            test_historical_neg_src_ood_mask, ~test_historical_neg_dst_ood_mask
        ).sum()
        num_historical_neg_id_id = np.logical_and(
            ~test_historical_neg_src_ood_mask, ~test_historical_neg_dst_ood_mask
        ).sum()
        num_historical_neg_dict["ood_ood"].append(num_historical_neg_ood_ood)
        num_historical_neg_dict["id_ood"].append(num_historical_neg_id_ood)
        num_historical_neg_dict["ood_id"].append(num_historical_neg_ood_id)
        num_historical_neg_dict["id_id"].append(num_historical_neg_id_id)

        num_inductive_neg_ood_ood = np.logical_and(
            test_inductive_neg_src_ood_mask, test_inductive_neg_dst_ood_mask
        ).sum()
        num_inductive_neg_id_ood = np.logical_and(
            ~test_inductive_neg_src_ood_mask, test_inductive_neg_dst_ood_mask
        ).sum()
        num_inductive_neg_ood_id = np.logical_and(
            test_inductive_neg_src_ood_mask, ~test_inductive_neg_dst_ood_mask
        ).sum()
        num_inductive_neg_id_id = np.logical_and(
            ~test_inductive_neg_src_ood_mask, ~test_inductive_neg_dst_ood_mask
        ).sum()
        num_inductive_neg_dict["ood_ood"].append(num_inductive_neg_ood_ood)
        num_inductive_neg_dict["id_ood"].append(num_inductive_neg_id_ood)
        num_inductive_neg_dict["ood_id"].append(num_inductive_neg_ood_id)
        num_inductive_neg_dict["id_id"].append(num_inductive_neg_id_id)

        # Calculate the average scores on test edges aggregated by (OOD, OOD), (ID, OOD), (OOD, ID), (ID, ID)
        # where the first element represents whether source node is in distribution and the second element represents whether destination node is in distribution

        score_pos_ood_ood = test_random_length_analysis["pos"]["scores"][
            np.logical_and(test_pos_src_ood_mask, test_pos_dst_ood_mask)
        ]
        score_pos_id_ood = test_random_length_analysis["pos"]["scores"][
            np.logical_and(~test_pos_src_ood_mask, test_pos_dst_ood_mask)
        ]
        score_pos_ood_id = test_random_length_analysis["pos"]["scores"][
            np.logical_and(test_pos_src_ood_mask, ~test_pos_dst_ood_mask)
        ]
        score_pos_id_id = test_random_length_analysis["pos"]["scores"][
            np.logical_and(~test_pos_src_ood_mask, ~test_pos_dst_ood_mask)
        ]
        # avg_score_pos_ood_ood = np.mean(test_random_length_analysis["pos"]["scores"][np.logical_and(test_pos_src_ood_mask, test_pos_dst_ood_mask)])
        # avg_score_pos_id_ood = np.mean(test_random_length_analysis["pos"]["scores"][np.logical_and(~test_pos_src_ood_mask, test_pos_dst_ood_mask)])
        # avg_score_pos_ood_id = np.mean(test_random_length_analysis["pos"]["scores"][np.logical_and(test_pos_src_ood_mask, ~test_pos_dst_ood_mask)])
        # avg_score_pos_id_id = np.mean(test_random_length_analysis["pos"]["scores"][np.logical_and(~test_pos_src_ood_mask, ~test_pos_dst_ood_mask)])

        score_random_neg_ood_ood = test_random_length_analysis["neg"]["scores"][
            np.logical_and(test_random_neg_src_ood_mask, test_random_neg_dst_ood_mask)
        ]
        score_random_neg_id_ood = test_random_length_analysis["neg"]["scores"][
            np.logical_and(~test_random_neg_src_ood_mask, test_random_neg_dst_ood_mask)
        ]
        score_random_neg_ood_id = test_random_length_analysis["neg"]["scores"][
            np.logical_and(test_random_neg_src_ood_mask, ~test_random_neg_dst_ood_mask)
        ]
        score_random_neg_id_id = test_random_length_analysis["neg"]["scores"][
            np.logical_and(~test_random_neg_src_ood_mask, ~test_random_neg_dst_ood_mask)
        ]
        # avg_score_random_neg_ood_ood = np.mean(test_random_length_analysis["neg"]["scores"][np.logical_and(test_random_neg_src_ood_mask, test_random_neg_dst_ood_mask)])
        # avg_score_random_neg_id_ood = np.mean(test_random_length_analysis["neg"]["scores"][np.logical_and(~test_random_neg_src_ood_mask, test_random_neg_dst_ood_mask)])
        # avg_score_random_neg_ood_id = np.mean(test_random_length_analysis["neg"]["scores"][np.logical_and(test_random_neg_src_ood_mask, ~test_random_neg_dst_ood_mask)])
        # avg_score_random_neg_id_id = np.mean(test_random_length_analysis["neg"]["scores"][np.logical_and(~test_random_neg_src_ood_mask, ~test_random_neg_dst_ood_mask)])

        score_historical_neg_ood_ood = test_historical_length_analysis["neg"]["scores"][
            np.logical_and(test_historical_neg_src_ood_mask, test_historical_neg_dst_ood_mask)
        ]
        score_historical_neg_id_ood = test_historical_length_analysis["neg"]["scores"][
            np.logical_and(~test_historical_neg_src_ood_mask, test_historical_neg_dst_ood_mask)
        ]
        score_historical_neg_ood_id = test_historical_length_analysis["neg"]["scores"][
            np.logical_and(test_historical_neg_src_ood_mask, ~test_historical_neg_dst_ood_mask)
        ]
        score_historical_neg_id_id = test_historical_length_analysis["neg"]["scores"][
            np.logical_and(~test_historical_neg_src_ood_mask, ~test_historical_neg_dst_ood_mask)
        ]
        # avg_score_historical_neg_ood_ood = np.mean(test_historical_length_analysis["neg"]["scores"][np.logical_and(test_historical_neg_src_ood_mask, test_historical_neg_dst_ood_mask)])
        # avg_score_historical_neg_id_ood = np.mean(test_historical_length_analysis["neg"]["scores"][np.logical_and(~test_historical_neg_src_ood_mask, test_historical_neg_dst_ood_mask)])
        # avg_score_historical_neg_ood_id = np.mean(test_historical_length_analysis["neg"]["scores"][np.logical_and(test_historical_neg_src_ood_mask, ~test_historical_neg_dst_ood_mask)])
        # avg_score_historical_neg_id_id = np.mean(test_historical_length_analysis["neg"]["scores"][np.logical_and(~test_historical_neg_src_ood_mask, ~test_historical_neg_dst_ood_mask)])

        score_inductive_neg_ood_ood = test_inductive_length_analysis["neg"]["scores"][
            np.logical_and(test_inductive_neg_src_ood_mask, test_inductive_neg_dst_ood_mask)
        ]
        score_inductive_neg_id_ood = test_inductive_length_analysis["neg"]["scores"][
            np.logical_and(~test_inductive_neg_src_ood_mask, test_inductive_neg_dst_ood_mask)
        ]
        score_inductive_neg_ood_id = test_inductive_length_analysis["neg"]["scores"][
            np.logical_and(test_inductive_neg_src_ood_mask, ~test_inductive_neg_dst_ood_mask)
        ]
        score_inductive_neg_id_id = test_inductive_length_analysis["neg"]["scores"][
            np.logical_and(~test_inductive_neg_src_ood_mask, ~test_inductive_neg_dst_ood_mask)
        ]
        # avg_score_inductive_neg_ood_ood = np.mean(test_inductive_length_analysis["neg"]["scores"][np.logical_and(test_inductive_neg_src_ood_mask, test_inductive_neg_dst_ood_mask)])
        # avg_score_inductive_neg_id_ood = np.mean(test_inductive_length_analysis["neg"]["scores"][np.logical_and(~test_inductive_neg_src_ood_mask, test_inductive_neg_dst_ood_mask)])
        # avg_score_inductive_neg_ood_id = np.mean(test_inductive_length_analysis["neg"]["scores"][np.logical_and(test_inductive_neg_src_ood_mask, ~test_inductive_neg_dst_ood_mask)])
        # avg_score_inductive_neg_id_id = np.mean(test_inductive_length_analysis["neg"]["scores"][np.logical_and(~test_inductive_neg_src_ood_mask, ~test_inductive_neg_dst_ood_mask)])

        # avg_score_pos["ood_ood"].append(avg_score_pos_ood_ood)
        # avg_score_pos["id_ood"].append(avg_score_pos_id_ood)
        # avg_score_pos["ood_id"].append(avg_score_pos_ood_id)
        # avg_score_pos["id_id"].append(avg_score_pos_id_id)

        # avg_score_random_neg["ood_ood"].append(avg_score_random_neg_ood_ood)
        # avg_score_random_neg["id_ood"].append(avg_score_random_neg_id_ood)
        # avg_score_random_neg["ood_id"].append(avg_score_random_neg_ood_id)
        # avg_score_random_neg["id_id"].append(avg_score_random_neg_id_id)

        # avg_score_historical_neg["ood_ood"].append(avg_score_historical_neg_ood_ood)
        # avg_score_historical_neg["id_ood"].append(avg_score_historical_neg_id_ood)
        # avg_score_historical_neg["ood_id"].append(avg_score_historical_neg_ood_id)
        # avg_score_historical_neg["id_id"].append(avg_score_historical_neg_id_id)

        # avg_score_inductive_neg["ood_ood"].append(avg_score_inductive_neg_ood_ood)
        # avg_score_inductive_neg["id_ood"].append(avg_score_inductive_neg_id_ood)
        # avg_score_inductive_neg["ood_id"].append(avg_score_inductive_neg_ood_id)
        # avg_score_inductive_neg["id_id"].append(avg_score_inductive_neg_id_id)
        if num_pos_ood_ood > 0 and num_random_neg_ood_ood > 0:
            random_ap_dict["ood_ood"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_ood_ood), np.zeros(num_random_neg_ood_ood)]),
                    np.concatenate([score_pos_ood_ood, score_random_neg_ood_ood]),
                )
            )
        if num_pos_id_ood > 0 and num_random_neg_id_ood > 0:
            random_ap_dict["id_ood"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_id_ood), np.zeros(num_random_neg_id_ood)]),
                    np.concatenate([score_pos_id_ood, score_random_neg_id_ood]),
                )
            )
        if num_pos_ood_id > 0 and num_random_neg_ood_id > 0:
            random_ap_dict["ood_id"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_ood_id), np.zeros(num_random_neg_ood_id)]),
                    np.concatenate([score_pos_ood_id, score_random_neg_ood_id]),
                )
            )
        if num_pos_id_id > 0 and num_random_neg_id_id > 0:
            random_ap_dict["id_id"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_id_id), np.zeros(num_random_neg_id_id)]),
                    np.concatenate([score_pos_id_id, score_random_neg_id_id]),
                )
            )

        if num_pos_ood_ood > 0 and num_historical_neg_ood_ood > 0:
            historical_ap_dict["ood_ood"].append(
                average_precision_score(
                    np.concatenate(
                        [np.ones(num_pos_ood_ood), np.zeros(num_historical_neg_ood_ood)]
                    ),
                    np.concatenate([score_pos_ood_ood, score_historical_neg_ood_ood]),
                )
            )
        if num_pos_id_ood > 0 and num_historical_neg_id_ood > 0:
            historical_ap_dict["id_ood"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_id_ood), np.zeros(num_historical_neg_id_ood)]),
                    np.concatenate([score_pos_id_ood, score_historical_neg_id_ood]),
                )
            )
        if num_pos_ood_id > 0 and num_historical_neg_ood_id > 0:
            historical_ap_dict["ood_id"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_ood_id), np.zeros(num_historical_neg_ood_id)]),
                    np.concatenate([score_pos_ood_id, score_historical_neg_ood_id]),
                )
            )
        if num_pos_id_id > 0 and num_historical_neg_id_id > 0:
            historical_ap_dict["id_id"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_id_id), np.zeros(num_historical_neg_id_id)]),
                    np.concatenate([score_pos_id_id, score_historical_neg_id_id]),
                )
            )

        if num_pos_ood_ood > 0 and num_inductive_neg_ood_ood > 0:
            inductive_ap_dict["ood_ood"].append(
                average_precision_score(
                    np.concatenate(
                        [np.ones(num_pos_ood_ood), np.zeros(num_inductive_neg_ood_ood)]
                    ),
                    np.concatenate([score_pos_ood_ood, score_inductive_neg_ood_ood]),
                )
            )
        if num_pos_id_ood > 0 and num_inductive_neg_id_ood > 0:
            inductive_ap_dict["id_ood"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_id_ood), np.zeros(num_inductive_neg_id_ood)]),
                    np.concatenate([score_pos_id_ood, score_inductive_neg_id_ood]),
                )
            )
        if num_pos_ood_id > 0 and num_inductive_neg_ood_id > 0:
            inductive_ap_dict["ood_id"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_ood_id), np.zeros(num_inductive_neg_ood_id)]),
                    np.concatenate([score_pos_ood_id, score_inductive_neg_ood_id]),
                )
            )
        if num_pos_id_id > 0 and num_inductive_neg_id_id > 0:
            inductive_ap_dict["id_id"].append(
                average_precision_score(
                    np.concatenate([np.ones(num_pos_id_id), np.zeros(num_inductive_neg_id_id)]),
                    np.concatenate([score_pos_id_id, score_inductive_neg_id_id]),
                )
            )

    print("\n")
    print(
        f"# test positive edges having an OOD src node and an OOD dst node: {np.mean(num_pos_dict['ood_ood'])} +/- {np.std(num_pos_dict['ood_ood'])}"
    )
    print(
        f"# test positive edges having an ID src node and an OOD dst node: {np.mean(num_pos_dict['id_ood'])} +/- {np.std(num_pos_dict['id_ood'])}"
    )
    print(
        f"# test positive edges having an OOD src node and an ID dst node: {np.mean(num_pos_dict['ood_id'])} +/- {np.std(num_pos_dict['ood_id'])}"
    )
    print(
        f"# test positive edges having an ID src node and an ID dst node: {np.mean(num_pos_dict['id_id'])} +/- {np.std(num_pos_dict['id_id'])}"
    )

    # print("Positive test edge avg/std scores aggregated by (OOD, OOD), (ID, OOD), (OOD, ID), (ID, ID):")
    # for key in avg_score_pos.keys():
    #     print(f"{key}: {round(np.mean(avg_score_pos[key]), 4)} +/- {round(np.std(avg_score_pos[key]), 4)}")

    print("\n")
    print(
        f"# random negative test edges having an OOD src node and an OOD dst node: {np.mean(num_random_neg_dict['ood_ood'])} +/- {np.std(num_random_neg_dict['ood_ood'])}"
    )
    print(
        f"# random negative test edges having an ID src node and an OOD dst node: {np.mean(num_random_neg_dict['id_ood'])} +/- {np.std(num_random_neg_dict['id_ood'])}"
    )
    print(
        f"# random negative test edges having an OOD src node and an ID dst node: {np.mean(num_random_neg_dict['ood_id'])} +/- {np.std(num_random_neg_dict['ood_id'])}"
    )
    print(
        f"# random negative test edges having an ID src node and an ID dst node: {np.mean(num_random_neg_dict['id_id'])} +/- {np.std(num_random_neg_dict['id_id'])}"
    )
    print(
        "Random avererage precision scores (avg/std) aggregated by (OOD, OOD), (ID, OOD), (OOD, ID), (ID, ID):"
    )
    for key in random_ap_dict.keys():
        if len(random_ap_dict[key]) == 0:
            continue
        gap_wrt_id = np.array(random_ap_dict["id_id"]) - np.array(random_ap_dict[key])
        print(
            f"{key}: {round(np.mean(random_ap_dict[key]), 4)} +/- {round(np.std(random_ap_dict[key]), 4)} ({round(np.mean(gap_wrt_id), 4)} +/- {round(np.std(gap_wrt_id), 4)})"
        )

    print("\n")
    print(
        f"# historical negative test edges having an OOD src node and an OOD dst node: {np.mean(num_historical_neg_dict['ood_ood'])} +/- {np.std(num_historical_neg_dict['ood_ood'])}"
    )
    print(
        f"# historical negative test edges having an ID src node and an OOD dst node: {np.mean(num_historical_neg_dict['id_ood'])} +/- {np.std(num_historical_neg_dict['id_ood'])}"
    )
    print(
        f"# historical negative test edges having an OOD src node and an ID dst node: {np.mean(num_historical_neg_dict['ood_id'])} +/- {np.std(num_historical_neg_dict['ood_id'])}"
    )
    print(
        f"# historical negative test edges having an ID src node and an ID dst node: {np.mean(num_historical_neg_dict['id_id'])} +/- {np.std(num_historical_neg_dict['id_id'])}"
    )
    print(
        "Historical avererage precision scores (avg/std) aggregated by (OOD, OOD), (ID, OOD), (OOD, ID), (ID, ID):"
    )
    for key in historical_ap_dict.keys():
        if len(historical_ap_dict[key]) == 0:
            continue
        gap_wrt_id = np.array(historical_ap_dict["id_id"]) - np.array(historical_ap_dict[key])
        print(
            f"{key}: {round(np.mean(historical_ap_dict[key]), 4)} +/- {round(np.std(historical_ap_dict[key]), 4)} ({round(np.mean(gap_wrt_id), 4)} +/- {round(np.std(gap_wrt_id), 4)})"
        )

    print("\n")
    print(
        f"# inductive negative test edges having an OOD src node and an OOD dst node: {np.mean(num_inductive_neg_dict['ood_ood'])} +/- {np.std(num_inductive_neg_dict['ood_ood'])}"
    )
    print(
        f"# inductive negative test edges having an ID src node and an OOD dst node: {np.mean(num_inductive_neg_dict['id_ood'])} +/- {np.std(num_inductive_neg_dict['id_ood'])}"
    )
    print(
        f"# inductive negative test edges having an OOD src node and an ID dst node: {np.mean(num_inductive_neg_dict['ood_id'])} +/- {np.std(num_inductive_neg_dict['ood_id'])}"
    )
    print(
        f"# inductive negative test edges having an ID src node and an ID dst node: {np.mean(num_inductive_neg_dict['id_id'])} +/- {np.std(num_inductive_neg_dict['id_id'])}"
    )
    print(
        "Inductive avererage precision scores (avg/std) aggregated by (OOD, OOD), (ID, OOD), (OOD, ID), (ID, ID):"
    )
    for key in inductive_ap_dict.keys():
        if len(inductive_ap_dict[key]) == 0:
            continue
        gap_wrt_id = np.array(inductive_ap_dict["id_id"]) - np.array(inductive_ap_dict[key])
        print(
            f"{key}: {round(np.mean(inductive_ap_dict[key]), 4)} +/- {round(np.std(inductive_ap_dict[key]), 4)} ({round(np.mean(gap_wrt_id), 4)} +/- {round(np.std(gap_wrt_id), 4)})"
        )


def main(args):
    """Load length analysis results from DyGFormer and DyGDecoder checkpoints and compare the
    average precision scores under ID/OOD settings for source and destination nodes in the test
    sets."""
    train_random_length_analysis_list = []
    test_random_length_analysis_list = []
    test_historical_length_analysis_list = []
    test_inductive_length_analysis_list = []
    if "multiruns" in args.ckpt_log_dir:
        runs = [
            obj
            for obj in os.listdir(args.ckpt_log_dir)
            if os.path.isdir(os.path.join(args.ckpt_log_dir, obj))
        ]
        for run in runs:
            train_random_length_analysis_path = os.path.join(
                args.ckpt_log_dir, run, "checkpoints", "train_random_length_analysis.pt"
            )
            test_random_length_analysis_path = os.path.join(
                args.ckpt_log_dir, run, "checkpoints", "test_random_length_analysis.pt"
            )
            test_historical_length_analysis_path = os.path.join(
                args.ckpt_log_dir, run, "checkpoints", "test_historical_length_analysis.pt"
            )
            test_inductive_length_analysis_path = os.path.join(
                args.ckpt_log_dir, run, "checkpoints", "test_inductive_length_analysis.pt"
            )
            train_random_length_analysis_list.append(torch.load(train_random_length_analysis_path))
            test_random_length_analysis_list.append(torch.load(test_random_length_analysis_path))
            test_historical_length_analysis_list.append(
                torch.load(test_historical_length_analysis_path)
            )
            test_inductive_length_analysis_list.append(
                torch.load(test_inductive_length_analysis_path)
            )
    else:
        train_random_length_analysis_path = os.path.join(
            args.ckpt_log_dir, "checkpoints", "train_random_length_analysis.pt"
        )
        test_random_length_analysis_path = os.path.join(
            args.ckpt_log_dir, "checkpoints", "test_random_length_analysis.pt"
        )
        test_historical_length_analysis_path = os.path.join(
            args.ckpt_log_dir, "checkpoints", "test_historical_length_analysis.pt"
        )
        test_inductive_length_analysis_path = os.path.join(
            args.ckpt_log_dir, "checkpoints", "test_inductive_length_analysis.pt"
        )
        train_random_length_analysis_list.append(torch.load(train_random_length_analysis_path))
        test_random_length_analysis_list.append(torch.load(test_random_length_analysis_path))
        test_historical_length_analysis_list.append(
            torch.load(test_historical_length_analysis_path)
        )
        test_inductive_length_analysis_list.append(torch.load(test_inductive_length_analysis_path))

    compare_id_ood_scores(
        train_random_length_analysis_list,
        test_random_length_analysis_list,
        test_historical_length_analysis_list,
        test_inductive_length_analysis_list,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# train_random_length_analysis = torch.load(args.train_random_length_analysis_path)
# test_random_length_analysis = torch.load(args.test_random_length_analysis_path)
# test_historical_length_analysis = torch.load(args.test_historical_length_analysis_path)
# test_inductive_length_analysis = torch.load(args.test_inductive_length_analysis_path)

# # Find the largest seen time diffs in the training sets for both source and destination nodes
# src_max_time_diff, dst_max_time_diff = np.nanmax(train_random_length_analysis["pos"]["src"]["max_time_diffs"]), np.nanmax(train_random_length_analysis["pos"]["dst"]["max_time_diffs"])

# # Create a mask for test samples that contain time diffs larger than the largest seen time diffs in the training sets, call them OOD samples
# pos_src_ood_mask = test_random_length_analysis["pos"]["src"]["max_time_diffs"] > src_max_time_diff
# pos_dst_ood_mask = test_random_length_analysis["pos"]["dst"]["max_time_diffs"] > dst_max_time_diff

# neg_src_ood_mask = test_random_length_analysis["neg"]["src"]["max_time_diffs"] > src_max_time_diff
# neg_dst_ood_mask = test_random_length_analysis["neg"]["dst"]["max_time_diffs"] > dst_max_time_diff

# # Calculate the number of ID/OOD samples in the test sets
# print(f"# test edges having an OOD src node and an OOD dst node: {np.logical_and(pos_src_ood_mask, pos_dst_ood_mask).sum()}")
# print(f"# test edges having an ID src node and an OOD dst node: {np.logical_and(~pos_src_ood_mask, pos_dst_ood_mask).sum()}")
# print(f"# test edges having an OOD src node and an ID dst node: {np.logical_and(pos_src_ood_mask, ~pos_dst_ood_mask).sum()}")
# print(f"# test edges having an ID src node and an ID dst node: {np.logical_and(~pos_src_ood_mask, ~pos_dst_ood_mask).sum()}")

# print("\n")

# print("Scores on positive test edges having an OOD src node and an OOD dst node:", np.nanmean(test_random_length_analysis["pos"]["scores"][np.logical_and(pos_src_ood_mask, pos_dst_ood_mask)]))
# print("Scores on positive test edges having an ID src node and an OOD dst node:", np.nanmean(test_random_length_analysis["pos"]["scores"][np.logical_and(~pos_src_ood_mask, pos_dst_ood_mask)]))
# print("Scores on positive test edges having an OOD src node and an ID dst node:", np.nanmean(test_random_length_analysis["pos"]["scores"][np.logical_and(pos_src_ood_mask, ~pos_dst_ood_mask)]))
# print("Scores on positive test edges having an ID src node and an ID dst node:", np.nanmean(test_random_length_analysis["pos"]["scores"][np.logical_and(~pos_src_ood_mask, ~pos_dst_ood_mask)]))

# print("\n")

# print("Scores on random negative test edges having an OOD src node and an OOD dst node:", np.nanmean(test_random_length_analysis["neg"]["scores"][np.logical_and(neg_src_ood_mask, neg_dst_ood_mask)]))
# print("Scores on random negative test edges having an ID src node and an OOD dst node:", np.nanmean(test_random_length_analysis["neg"]["scores"][np.logical_and(~neg_src_ood_mask, neg_dst_ood_mask)]))
# print("Scores on random negative test edges having an OOD src node and an ID dst node:", np.nanmean(test_random_length_analysis["neg"]["scores"][np.logical_and(neg_src_ood_mask, ~neg_dst_ood_mask)]))
# print("Scores on random negative test edges having an ID src node and an ID dst node:", np.nanmean(test_random_length_analysis["neg"]["scores"][np.logical_and(~neg_src_ood_mask, ~neg_dst_ood_mask)]))

# print("\n")
