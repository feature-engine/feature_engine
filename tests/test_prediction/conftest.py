import pandas as pd
import pytest


@pytest.fixture(scope="module")
def df_pred():
    data = {
        "Name": [
            "tom",
            "nick",
            "krish",
            "megan",
            "peter",
            "jordan",
            "fred",
            "sam",
            "alexa",
            "brittany"
        ],
        "City": [
            "London",
            "Manchester",
            "Liverpool",
            "Bristol",
            "Manchester",
            "Liverpool",
            "London",
            "Liverpool",
            "Manchester",
            "London"],
        "Studies": [
            "Bachelor",
            "Bachelor",
            "PhD",
            "Masters",
            "Bachelor",
            "PhD",
            "None",
            "Masters",
            "Masters",
            "Bachelor"
        ],
        "Age": [20, 44, 19, 33, 51, 40, 41, 37, 30, 54],
        "Marks": [1.0, 0.8, 0.6, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.2],
        "Plays_Football": [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_pred_small():
    data = {
        "Names": ["louis", "michael", "arnold", "joyce", "claire", "allison"],
        "City": [
            "London",
            "London",
            "Liverpool",
            "Manchester",
            "Bristol",
            "Manchester"
        ],
        "Studies": ["Masters", "Bachelor", "Bachelor", "PhD", "Phd", "Masters"],
        "Age": [29, 39, 49, 25, 35, 55],
        "Marks": [0.9, 0.6, 0.1, 0.5, 0.3, 0.8],
        "Plays_Football": [0, 1, 1, 0, 1, 1],
    }

    df = pd.DataFrame(data)

    return df
