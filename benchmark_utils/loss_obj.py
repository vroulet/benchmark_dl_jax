from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from clu import metrics as clu_metrics
    from flax import struct
    from flax.training import common_utils
    from jax import numpy as jnp
    import optax


def cross_entropy_fun(logits, labels):
    num_classes = logits.shape[-1]
    labels_ = common_utils.onehot(labels, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits, labels_)
    return jnp.mean(loss)


@struct.dataclass
class EvalMetrics(clu_metrics.Collection):
    accuracy: clu_metrics.Accuracy
    loss: clu_metrics.Average.from_output("loss")
