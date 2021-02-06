import matplotlib

from EnergyTransferResults import EnergyTransferResults


def test_energy_transfer_results():
    """Check energy transfer results are calculated or loaded correctly.

    Returns
    -------

    """
    dt = 0.05  # s
    gamma_range = (1e-2, 100)
    eta12_range = (1e-3, 100)

    def update_x(args_dict, z):
        args_dict.update({"n12": z / dt})

    def update_y(args_dict, z):
        D1 = args_dict["D2"] * z
        args_dict.update({"D1": D1})

    en_tr_results = EnergyTransferResults(
        trials=2,
        verbose=True,
        cluster=True,
        resolution=0,
        Ms=(1000,),
        x_label=r"$\eta_{12}$",
        x_range=eta12_range,
        update_x=update_x,
        y_label=r"$\gamma \equiv D_1/D_2$",
        y_range=gamma_range,
        update_y=update_y,
        dt=dt,
        angle=0,
        rotation=True,
        print_MLE=False,
        title="n1={n1:.2f}, n2={n2:.2f}, D2={D2:.2f},\ndt={dt}, L0={L0:.2f}",
    )

    en_tr_results.run()


def test_check_mle(caplog):
    """Test that `check_mle` option functions as expected.

    Returns
    -------

    """
    matplotlib.use("Agg")
    dt = 0.05  # s
    gamma_range = (1e-2, 100)
    eta12_range = (1e-3, 100)

    def update_x(args_dict, z):
        args_dict.update({"n12": z / dt})

    def update_y(args_dict, z):
        D1 = args_dict["D2"] * z
        args_dict.update({"D1": D1})

    default_args = dict(
        trials=2,
        verbose=True,
        cluster=False,
        resolution=0,
        Ms=(10,),
        x_label=r"$\eta_{12}$",
        x_range=eta12_range,
        update_x=update_x,
        y_label=r"$\gamma \equiv D_1/D_2$",
        y_range=gamma_range,
        update_y=update_y,
        dt=dt,
        angle=0,
        rotation=True,
        print_MLE=False,
        title="n1={n1:.2f}, n2={n2:.2f}, D2={D2:.2f},\ndt={dt}, L0={L0:.2f}",
    )

    # make sure previous results exist
    EnergyTransferResults(**default_args).run()

    EnergyTransferResults(**default_args, check_mle=True).run()
    assert any("MLE check" in rec.msg for rec in caplog.records)
