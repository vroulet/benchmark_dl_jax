from functools import partial

from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import jax
    import optax

    # import your reusable functions here


# FIXME: add weight decay
def obj_fun_template(params, batch_stats, x, y, net, loss_fun):
    logits, mutated_vars = net.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        mutable=["batch_stats"],
    )
    loss_val = loss_fun(logits, y)
    return loss_val, mutated_vars["batch_stats"]


def train_step_template(
    params, batch_stats, opt_state, x, y, obj_grad_fun, optimizer
):
    grads, batch_stats = obj_grad_fun(params, batch_stats, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, batch_stats, opt_state


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "OptaxSGD"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": [1e-3, 1e-4],
        "momentum": [0.0, 0.9],
        "nesterov": [False, True],
    }

    def set_objective(
        self, train_ds, test_ds, initialize_model, net, loss_fun
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.train_ds, self.test_ds = train_ds, test_ds
        self.initialize_model = initialize_model
        self.net, self.loss_fun = net, loss_fun

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.

        optimizer = optax.sgd(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        obj_fun = partial(
            obj_fun_template, net=self.net, loss_fun=self.loss_fun
        )
        obj_grad_fun = jax.grad(obj_fun, has_aux=True)
        params, batch_stats = self.initialize_model()
        opt_state = optimizer.init(params)

        train_step_ = partial(
            train_step_template, obj_grad_fun=obj_grad_fun, optimizer=optimizer
        )
        train_step = jax.jit(train_step_)
        for _ in range(n_iter):
            for x, y in self.train_ds:
                params, batch_stats, opt_state = train_step(
                    params, batch_stats, opt_state, x, y
                )

        self.params, self.batch_stats = params, batch_stats

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(params=self.params, batch_stats=self.batch_stats)
