# Quick tests
from datasets.cifar10 import Dataset
from objective import Objective
from solvers.optax_sgd import Solver


def test_pipeline():
    dataset = Dataset()
    dataset.batch_size = 1024
    ds = dataset.get_data()
    train_ds, test_ds, info_ds = ds["train_ds"], ds["test_ds"], ds["info_ds"]
    objective = Objective()
    objective.model = "resnet18"
    objective.loss = "cross_entropy"
    objective.set_data(train_ds, test_ds, info_ds)
    objective_ = objective.get_objective()
    train_ds, test_ds = objective_["train_ds"], objective_["test_ds"]
    initialize_model = objective_["initialize_model"]
    net, loss_fun = objective_["net"], objective_["loss_fun"]
    solver = Solver()
    solver.learning_rate, solver.momentum, solver.nesterov = 1e-3, 1.0, False
    solver.set_objective(train_ds, test_ds, initialize_model, net, loss_fun)
    solver.run(n_iter=2)
    params, batch_stats = solver.get_result()
    results = objective.compute(params, batch_stats)
    assert 'value' in results.keys()


if __name__ == '__main__':
    test_pipeline()

    