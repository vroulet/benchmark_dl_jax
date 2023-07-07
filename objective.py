from functools import partial
from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import jax
    from jax import numpy as jnp
    from benchmark_utils import (
        ResNet,
        ResNetBlock,
        BottleneckResNetBlock,
        cross_entropy_fun,
        EvalMetrics,
    )


# FIXME make it work without mutable and with other mutable (use e.g. kwargs_net)
def eval_step_template(params, batch_stats, x, y, metrics, net, loss_fun):
    logits = net.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        train=False,
        mutable=False,
    )
    loss_val = loss_fun(logits, y)
    new_metrics = EvalMetrics.single_from_model_output(
        logits=logits, labels=y, loss=loss_val
    )
    if metrics is None:
        metrics = new_metrics
    else:
        metrics = metrics.merge(new_metrics)
    return metrics


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "ImageClassification"

    requirements = ["pip:jax", "pip:jaxlib", "pip:optax", "pip:clu", "pip:flax"]

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        "model": ["resnet18", "resnet34", "resnet50"],
        "loss": ["cross_entropy"],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, train_ds, test_ds, info_ds):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.train_ds, self.test_ds, self.info_ds = train_ds, test_ds, info_ds

        self.net, self.loss_fun = self.set_model()

        eval_step = partial(
            eval_step_template, net=self.net, loss_fun=self.loss_fun
        )
        self.eval_step = jax.jit(eval_step)
        # `set_data` can be used to preprocess the data.

    def set_model(self):
        if self.model == "resnet18":
            net_arch = partial(
                ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock
            )
        elif self.model_size == "resnet34":
            net_arch = partial(
                ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock
            )
        elif self.model_size == "resnet50":
            net_arch = partial(
                ResNet,
                stage_sizes=[3, 4, 6, 3],
                block_cls=BottleneckResNetBlock,
            )
        else:
            raise NotImplementedError
        # FIXME: vary seed as option?
        net = net_arch(num_classes=self.info_ds.num_classes)
        if self.loss == "cross_entropy":
            loss_fun = cross_entropy_fun
        else:
            raise NotImplementedError
        return net, loss_fun

    def initialize_model(self):
        seed = 0
        rng = jax.random.PRNGKey(seed)
        variables = self.net.init(rng, jnp.ones((1, *self.info_ds.input_shape)))
        params, batch_stats = variables["params"], variables["batch_stats"]
        return params, batch_stats

    def eval(self, params, batch_stats, ds):
        metrics = None
        for x, y in ds:
            metrics = self.eval_step(params, batch_stats, x, y, metrics)
        return metrics.compute()

    def compute(self, output):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        params, batch_stats = output['params'], output['batch_stats']
        train_metrics = self.eval(params, batch_stats, self.train_ds)
        test_metrics = self.eval(params, batch_stats, self.test_ds)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=train_metrics["loss"].item(),
            test_loss=test_metrics["loss"].item(),
            train_accuracy=train_metrics["accuracy"].item(),
            test_accuracy=test_metrics["accuracy"].item(),
        )

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return self.initialize_model()

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        return dict(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            initialize_model=self.initialize_model,
            net=self.net,
            loss_fun=self.loss_fun,
        )
