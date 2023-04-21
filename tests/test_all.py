from tnsu.examples import afh_peps_ground_state_experiment


def test_alg1():
    _, energies = afh_peps_ground_state_experiment(size=4, bc="obc", d_max_=[3], error=1e-6,
                                                   max_iterations=200, dts=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                                   h_k=0., plot_results=False, save_network=False)
    assert abs(energies[0] - -0.5354082561993647) < 1e-13


