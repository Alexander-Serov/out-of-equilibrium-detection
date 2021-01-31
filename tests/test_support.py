import copy

import numpy as np

from constants_main import (
    FLOAT_FORMAT,
    MLE_GUESSES_FOLDER,
    MLE_GUESSES_TO_KEEP,
    data_folder,
)
from support import get_mle_file, load_mle_guesses, save_mle_guess

true_params = {
    "model": "test_model",
    "link": True,
    "D1": 3 / 7,
    "D2": 500 / 6,
    "n1": 37,
    "n2": 3.7584583939,
    "n12": 5e-7,
}
ignore_keys = {"model", "link"}


def test_get_mle_file():

    file = get_mle_file(true_params)
    filename = file.name

    assert filename.endswith(".json")

    assert "model" in filename and true_params["model"] in filename
    assert "link" in filename and str(true_params["link"]) in filename

    for key in set(true_params.keys()) - ignore_keys:
        assert f"_{key}=(" in filename
        assert f"{true_params[key]:{FLOAT_FORMAT}}" in filename


def test_save_mle_guess():
    values = -7.66 + np.arange(MLE_GUESSES_TO_KEEP * 2)
    mle = copy.copy(true_params)
    for key in ignore_keys:
        mle.pop(key)

    filename = get_mle_file(true_params)
    filepath = data_folder / MLE_GUESSES_FOLDER / filename

    # -
    # Save several MLE guesses, but below the max. keep number
    try:
        filepath.unlink()
    except FileNotFoundError:
        pass

    values_to_save = values[: MLE_GUESSES_TO_KEEP - 1]
    for value in values_to_save:
        save_mle_guess(true_params=true_params, mle=mle, value=value)

    # Check correctly found
    loaded = load_mle_guesses(true_params=true_params)
    expected = [[v, mle] for v in values_to_save]
    assert loaded == expected

    # -
    # Try appending to same file
    value = 100
    save_mle_guess(true_params=true_params, mle=mle, value=value)

    # Check correctly found
    loaded = load_mle_guesses(true_params=true_params)
    expected = expected + [[value, mle]]
    assert loaded == expected

    # -
    # Save several MLE guesses, above the max. keep number
    try:
        filepath.unlink()
    except FileNotFoundError:
        pass

    values_to_save = values[: MLE_GUESSES_TO_KEEP + 1]
    for value in values_to_save:
        save_mle_guess(true_params=true_params, mle=mle, value=value)

    # Check correctly found
    loaded = load_mle_guesses(true_params=true_params)
    expected = [[v, mle] for v in values_to_save[:MLE_GUESSES_TO_KEEP]]
    assert loaded == expected
