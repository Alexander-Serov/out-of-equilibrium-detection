from support import get_mle_file
from trajectory import Trajectory


def test_get_mle(monkeypatch, capsys):
    kwargs = dict(M=4, recalculate_BF=True)
    traj = Trajectory(**kwargs)

    true_params = dict(traj.true_params, link=True)
    guess_file = get_mle_file(true_params)
    try:
        guess_file.unlink()
    except FileNotFoundError:
        pass

    msg = "MLE guess loaded successfully"
    for i in range(2):
        capsys.readouterr()
        assert traj.MLE_link is not None and traj.MLE_link
        assert (msg in capsys.readouterr().out) == bool(i)
        traj = Trajectory(**kwargs)
