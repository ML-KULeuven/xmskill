"""Code to simulate the GAX of a player.

Performs MCMC to compute distributions of GAX for a variety of skill levels
and number of shot attempts.

:author: Jesse Davis and Pieter Robberechts
:copyright: Copyright 2023 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import xgfeatures as xgfeat


def season_shots(num_shots, samples, skill, grid, footed, headed, other, seed=1234):
    """Simulate the GAX for a player that takes a hypothetical number of shots.

    The shots are sampled from empirical location and body part distributions.
    The xG is computed according to a trained model. Whether a goal is scored
    is sampled from the xG distribution. This xG distribution can be biased,
    i.e., upweighted or downweighted to simulate a player being a better or
    worse finisher than average.

    Parameters
    ----------
    num_shots : int
        The number of shots taken by the player.
    samples : int
        The number of times that `num_shots` are drawn during the MCMC
        procedure.
    skill : float
        The skill level of the player wrt the baseline xG model. The xG is
        multiplied by this value.
    grid : 1D array
        The distribution of shots per location.
    footed : 1D array
        The percentage of footed shots in each location.
    headed : 1D array
        The percentage of headed shots in each location.
    other : 1D array
        The percentage of other shots in each location.
    seed : int
        The seed to use for the random number generator for reproducibility.

    Returns
    -------
    DataFrame
        A DataFrame with the GAX, number of goals, and cumulative xG for each
        sample.
    """
    np.random.seed(seed)
    season_totals = []
    for _ in tqdm(range(samples)):
        # sample the number of shots taken from each location
        shots_per_loc = np.random.multinomial(num_shots, grid)
        sum_goals = 0
        sum_xg = 0
        for i in range(shots_per_loc.size):
            # get the appropiate x,y coordinates for the grid cell
            y_start = i // 105
            x_start = i - (y_start * 105)

            # construct the distribution of body parts used to shoot from this location
            body_dist = [footed[i], headed[i], other[i]]
            sum_body_dist = footed[i] + headed[i] + other[i]
            if shots_per_loc[i] > 0 and (np.abs(sum_body_dist - 1) > 0.00001):
                warnings.warn(f"Unnormalized Distribution: {body_dist}")

            for _ in range(shots_per_loc[i]):
                # randomly generate position in the considered grid cell
                x_pos = x_start + np.random.uniform(0, 1, 1)
                y_pos = y_start + np.random.uniform(0, 1, 1)
                # randomly sample body part used for the shot
                bodyPartUsed = np.random.multinomial(1, body_dist)
                # create feature vector for shot's true position
                trueShotEx = [
                    x_pos,
                    y_pos,
                    xgfeat.dist_to_center_goal(x_pos, y_pos),
                    xgfeat.angle_to_goal(x_pos, y_pos),
                    bodyPartUsed[1],
                    bodyPartUsed[2],
                ]

                # determine the shot's label
                val = intercept + np.dot(trueShotEx, weightVector)
                prob = np.exp(val) / (1 + np.exp(val))
                sum_xg += prob[0]

                # modify xG by skill level
                prob = prob * skill
                if prob > 1:
                    prob = 0.99
                # flip weighted coin to see if the shots converts
                sampled_goal = np.random.uniform(0, 1, 1)
                if sampled_goal < prob:
                    sum_goals += 1

        # GAX, number of goals, cumulative xG
        season_totals.append(
            {"GAX": sum_goals - sum_xg, "Num Goals": sum_goals, "Cumulative xG": sum_xg}
        )

    return pd.DataFrame(season_totals)


if __name__ == "__main__":
    # intercept of logistic regression model
    intercept = 0  # to fill in with your trained model
    # weight for x, y, dist, angle, head, other
    weightVector = [0, 0, 0, 0, 0, 0]  # to fill in with your trained model


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

    shot_attempt = [25, 50, 75, 100, 125, 150]
    skill_level = [1.0, 1.05, 1.1, 1.15, 1.25]

    num_args = len(sys.argv)
    if num_args == 1:
        for i in range(len(shot_attempt)):
            for j in range(len(skill_level)):
                file = (
                    "season-shot-10k-"
                    + str(shot_attempt[i])
                    + "-skill"
                    + str(skill_level[j])
                    + ".csv"
                )
                df = season_shots(
                    shot_attempt[i], 10000, skill_level[j], grid, footed, headed, other
                )
                df.to_csv(file)

    elif num_args == 2:
        print("good shooter distributions")
        aDistribution = np.loadtxt("all_good_smoothed.csv", delimiter=",")
        agrid = aDistribution.flatten()
        sumprobs_agrid = np.sum(agrid)
        agrid = agrid / sumprobs_agrid
        for i in range(len(shot_attempt)):
            for j in range(len(skill_level)):
                file = (
                    "season-shot-10k-good-shooters-"
                    + str(shot_attempt[i])
                    + "-skill"
                    + str(skill_level[j])
                    + ".csv"
                )
                df = season_shots(
                    shot_attempt[i], 10000, skill_level[j], agrid, footed, headed, other
                )
                df.to_csv(file)

    elif num_args == 3:
        print("player distributions")
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
            pDistribution = np.loadtxt(
                "player_shot_distributions/" + playernames[dx] + "_smoothed.csv",
                delimiter=",",
            )
            pgrid = pDistribution.flatten()
            sumprobs_pgrid = np.sum(pgrid)
            pgrid = pgrid / sumprobs_pgrid
            for i in range(len(shot_attempt)):
                for j in range(len(skill_level)):
                    file = (
                        "season-shot-10k-"
                        + playernames[dx]
                        + "-"
                        + str(shot_attempt[i])
                        + "-skill"
                        + str(skill_level[j])
                        + ".csv"
                    )
                    df = season_shots(
                        shot_attempt[i], 10000, skill_level[j], pgrid, footed, headed, other
                    )
                    df.to_csv(file)

    else:
        print("wrong number of arguments")
