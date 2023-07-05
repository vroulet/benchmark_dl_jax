"""CIFAR10 dataset loading in tensorflow (could use other loaders)"""
import functools
from typing import NamedTuple, Sequence
from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import tensorflow as tf
    import tensorflow_datasets as tfds

class InfoDataset(NamedTuple):
    num_train: int
    num_test: int
    num_classes: int
    input_shape: Sequence[int]


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "CIFAR10"

    requirements = ['tensorflow', 'tensorflow-datasets']
    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {"batch_size": [128]}

    def _process_sample(self, x, y, mean_rgb, std_rgb):
        image = tf.cast(x, tf.float32)
        image = (image - mean_rgb) / std_rgb
        return x, y
    
    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data.

        mean_rgb = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
        stddev_rgb = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]
        mean_rgb = tf.constant(mean_rgb, shape=[
                               1, 1, 3], dtype=tf.float32)
        std_rgb = tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=tf.float32)

        process_sample = functools.partial(
            self._process_sample, mean_rgb=mean_rgb, std_rgb=std_rgb)

        train_ds, test_ds = tfds.load(
            'cifar10',
            split=['train', 'test'],
            as_supervised=True,
        )

        # FIXME: evenutally use train_ds.repeat() and shuffle regularly, same
        # for test_ds
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(50000)
        train_ds = train_ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(self.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        train_ds = tfds.as_numpy(train_ds)

        test_ds = test_ds.cache()
        test_ds = test_ds.shuffle(10000)
        test_ds = test_ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size, drop_remainder=True)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = tfds.as_numpy(test_ds)

        info_ds = InfoDataset(num_train=50000, num_test=10000,
                              num_classes=10, input_shape=(32, 32, 3))
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(train_ds=train_ds, test_ds=test_ds, info_ds=info_ds)
    

# # Quick test
# if __name__ == '__main__':
#     ds = Dataset()
#     ds.batch_size = 128
#     out = ds.get_data()
#     train_ds, test_ds, indo_ds = out['train_ds'], out['test_ds'], out['info_ds']
#     i = 0
#     for x, y in train_ds:
#         print(x.shape)
#         i += 1
#         if i > 2:
#             break