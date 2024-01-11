"""
:author: Jesse Davis and Pieter Robberechts
:copyright: Copyright 2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import warnings
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

import xgfeatures as xgfeat


def season_shots(skill_levels, shot_attempts, grid, footed, headed, other):
    """Generate a biased dataset of shots.

    Given a distribution of skill levels and number of shots attempted by
    a player from each skill level, generate a dataset of shots. The shots are
    sampled from empirical location and body part distributions. To simulate
    a player being a better or worse finisher than average, we upweight or
    downweight the probability of scoring a goal.

    Parameters
    ----------
    skill_levels : array
        An array with the skill level of each player.
    shot_attempts : array
        An array with the number of shots attempted by each player.
    grid : array
        The distribution of shots per location.
    footed : array
        The percentage of footed shots in each location.
    headed : array
        The percentage of headed shots in each location.
    other : array
        The percentage of other shots in each location.

    Returns
    -------
    DataFrame
        A DataFrame with the GAX, number of goals, and cumulative xG for each
        sample.
    """
    data = []
    for sid in range(len(skill_levels)):
        sum_shots = 0
        sum_goals = 0
        sum_true_xg = 0
        sum_skill_xg = 0
        # for sa in range(shot_attempts[sid]):
        shots_per_loc = np.random.multinomial(shot_attempts[sid], grid)
        for i in tqdm(range(shots_per_loc.size), desc="Genering shots for skill level " + str(sid)):
            y_start = i // 105
            x_start = i - (y_start * 105)
            body_dist = [footed[i], headed[i], other[i]]
            sum_body_dist = footed[i] + headed[i] + other[i]
            if shots_per_loc[i] > 0 and (np.abs(sum_body_dist - 1) > 0.00001):
                warnings.warn(f"Unnormalized distribution: {body_dist}")

            for _ in range(shots_per_loc[i]):
                sum_shots = sum_shots + 1
                # randomly generate position the grid
                x_pos = x_start + np.random.uniform(0, 1, 1)
                y_pos = y_start + np.random.uniform(0, 1, 1)

                body_part_used = np.random.multinomial(1, body_dist)
                # create feature vector for shot's true position
                true_shot_xg = [
                    x_pos,
                    y_pos,
                    xgfeat.dist_to_center_goal(x_pos, y_pos),
                    xgfeat.angle_to_goal(x_pos, y_pos),
                    body_part_used[1],
                    body_part_used[2],
                ]
                # determine shots label
                val = intercept + np.dot(true_shot_xg, weight_vector)
                true_xg = np.exp(val) / (1 + np.exp(val))
                # account for the skill
                prob = true_xg * skill_levels[sid]
                sum_true_xg += true_xg
                sum_skill_xg += prob
                if prob > 1:
                    prob = 0.99
                    # flip weighted coin to see if the shots converts
                sampled_goal = np.random.uniform(0, 1, 1)
                is_goal = 0
                if sampled_goal < prob:
                    is_goal = 1
                    sum_goals = sum_goals + 1
                data.append(
                    {
                        "skill level": skill_levels[sid],
                        "true xg": true_xg,
                        "skill adjusted xg": prob,
                        "x": x_pos,
                        "y": y_pos,
                        "dist": true_shot_xg[2],
                        "angle": true_shot_xg[3],
                        "foot": body_part_used[0],
                        "head": body_part_used[1],
                        "other": body_part_used[2],
                        "goal": is_goal,
                    }
                )
        logging.debug(
            "Sum Shots "
            + str(sum_shots)
            + " "
            + str(skill_levels[sid])
            + " goals "
            + str(sum_goals)
            + " xg "
            + str(sum_true_xg)
            + " skill xg "
            + str(sum_skill_xg)
        )

    return pd.DataFrame(data)


def biased_data(stem, grid, footed, headed, other):
    shot_numbers = np.array([250000, 250000, 250000, 250000, 250000], np.int32)
    skill = [1.05, 1.1, 1.15, 1.2, 1.25]
    df = season_shots(skill, shot_numbers, grid, footed, headed, other)
    df.to_csv(stem + "biased-generic-distribution.csv")
    aDistribution = np.loadtxt("all_good_smoothed.csv", delimiter=",")
    agrid = aDistribution.flatten()
    sumprobs_agrid = np.sum(agrid)
    agrid = agrid / sumprobs_agrid

    df = season_shots(skill, shot_numbers, agrid, footed, headed, other)
    df.to_csv(stem + "biased-good-shooter-distribution.csv")


def run_simulation_generic(stem, skill_levels, grid, footed, headed, other):
    shot_numbers = np.array(
        [
            [100000, 800000, 50000, 50000],
            [50000, 750000, 100000, 100000],
            [50000, 650000, 100000, 200000],
        ],
        np.int32,
    )
    file_ids = ["100k-800k-50k-50k", "50k-750k-100k-100k", "50k-650k-100k-200k"]
    for iter in range(len(file_ids)):
        print(file_ids[iter])
        df = season_shots(skill_levels, shot_numbers[iter], grid, footed, headed, other)
        df.to_csv(stem + file_ids[iter] + "-train.csv")
        df = season_shots(skill_levels, shot_numbers[iter], grid, footed, headed, other)
        df.to_csv(stem + file_ids[iter] + "-test.csv")


def run_simulation_players(skill_levels, footed, headed, other):
    shot_numbers = np.array([10000, 10000, 10000, 10000])
    playernames = [
        "Antoine Griezmann",
        "Harry Kane",
        "Karim Benzema",
        "Kevin De Bruyne",
        "Kylian Mbappe",
        "Lionel Messi",
        "Lorenzo Insigne",
        "Mohamed Salah",
        "Paul Pogba",
        "Robert Lewandowski",
        "Romelu Lukaku Menama",
        "Heung-Min Son",
    ]

    for dx in range(len(playernames)):
        distribution = np.loadtxt(
            "player_shot_distributions/" + playernames[dx] + "_smoothed.csv",
            delimiter=",",
        )
        grid = distribution.flatten()
        total = np.sum(grid)
        if total > 1:
            grid = grid / total
            print("Original Total " + str(total) + " " + playernames[dx])
        total = np.sum(grid)
        print("Total " + str(total))
        df = season_shots(skill_levels, shot_numbers, grid, footed, headed, other)
        df.to_csv(playernames[dx] + "-10k-smoothed.csv")


if __name__ == "__main__":

    # intercept, x, y, dist, angle, head, other]
    intercept = 14.301040398979099
    weight_vector = [
        -0.12903395599643944,
        0.0007081390917350903,
        -0.31351026346703825,
        0.09095528657471205,
        -1.2946488935455573,
        -0.19292432746094432,
    ]


    # Distribution of shots per location
    shotDistribution = np.loadtxt("shot_distribution.csv", delimiter=",")
    grid = shotDistribution.flatten()
    sumprobs_grid = np.sum(grid)
    grid = grid / sumprobs_grid

    # percentage of footed shots in each location
    footed_dist = np.loadtxt("shot_foot_distribution.csv", delimiter=",")
    footed = footed_dist.flatten()
    footed = np.nan_to_num(footed)


    # percentage of headed shots in each location
    headed_dist = np.loadtxt("shot_head_distribution.csv", delimiter=",")
    headed = headed_dist.flatten()
    headed = np.nan_to_num(headed)


    # percentage of other shots in each location
    other_dist = np.loadtxt("shot_other_distribution.csv", delimiter=",")
    other = other_dist.flatten()
    other = np.nan_to_num(other)


    skills_simple = [0.95, 1.0, 1.1, 1.2]

    np.random.seed(1234)
    num_args = len(sys.argv)
    if num_args == 1:
        print("Running player sim")
        run_simulation_players(skills_simple, footed, headed, other)
    elif num_args == 2:
        run_simulation_generic(
            "/cw/dtailocal/xmskill/MITSloan/xgskill-",
            skills_simple,
            grid,
            footed,
            headed,
            other,
        )
    elif num_args == 3:
        print("Running biased data")
        biased_data("/cw/dtailocal/xmskill/MITSloan/xgskill-", grid, footed, headed, other)
    else:
        print(
            "Two arguments needed: <file for train set> <file for test set> "
            + str(num_args)
        )
