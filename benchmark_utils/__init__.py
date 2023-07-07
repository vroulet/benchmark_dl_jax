# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax


from benchopt import safe_import_context  # pylint: disable=unused-import

from benchmark_utils.flax_net import ResNet, ResNetBlock, BottleneckResNetBlock  # pylint: disable=unused-import
from benchmark_utils.loss_obj import cross_entropy_fun, EvalMetrics  # pylint: disable=unused-import
