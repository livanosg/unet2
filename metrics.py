import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, AcceptableDTypes
from typeguard import typechecked


class DiceScore(tf.keras.metrics.Metric):
    """Computes F-Beta score.

    It is the weighted harmonic mean of precision
    and recall. Output range is [0, 1]. Works for
    both multi-class and multi-label classification.

    F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)

    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.

    Returns:
        F-Beta Score: float

    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].

        ValueError: If the `beta` value is less than or equal
        to 0.

    `average` parameter behavior:
        None: Scores for each class are returned

        micro: True positivies, false positives and
            false negatives are computed globally.

        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.

        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    @typechecked
    def __init__(
        self,
            num_classes: FloatTensorLike,
            average: str = None,
            name: str = "Dice_Score",
            dtype: AcceptableDTypes = None,
            **kwargs):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError("Unknown average type. Acceptable values "
                             "are: [None, micro, macro, weighted]")

        self.eps = 1e-20
        self.num_classes = num_classes
        self.average = average
        self.axis = 0
        self.init_shape = [self.num_classes]

        # if self.average != "micro":
        #     self.axis = 0
        #     self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.numerator = _zero_wt_init("numerator")
        self.denominator = _zero_wt_init("denominator")
        self.class_frequencies = _zero_wt_init("class_frequencies")

    # TODO: Add sample_weight support, currently it is
    # ignored during calculations.
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        self.numerator.assign_add(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=(0, 1, 2)))
        self.denominator.assign_add(tf.reduce_sum(tf.math.add(y_true, y_pred), axis=(0, 1, 2)))
        if self.average == 'weighted':
            self.class_frequencies.assign_add(tf.reduce_sum(y_true, axis=[0, 1, 2]))

    def result(self):
        # F1 micro
        if self.average == 'micro':
            f1 = tf.math.divide(tf.math.multiply(self.numerator, 2.0), tf.math.add(self.denominator, self.eps))
            f1 = tf.reduce_mean(f1)
        # F1 macro
        elif self.average == 'macro':
            f1 = tf.math.divide(2.0 * tf.reduce_sum(self.numerator), tf.reduce_sum(tf.math.add(self.denominator, self.eps)))
        # F1 weighted
        elif self.average == "weighted":
            weights = tf.math.divide(self.class_frequencies, tf.reduce_sum(self.class_frequencies))  # Correct F1 weighted
            f1 = tf.reduce_sum(tf.math.multiply(tf.math.divide(tf.math.multiply(self.numerator, 2.0), tf.math.add(self.denominator, self.eps)), weights))
        else:
            raise ValueError('wrong mode {}'.format(self.average))
        return f1

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
        }

        if self.threshold is not None:
            config["threshold"] = self.threshold

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        self.numerator.assign(tf.zeros(self.init_shape, self.dtype))
        self.denominator.assign(tf.zeros(self.init_shape, self.dtype))
        self.class_frequencies.assign(tf.zeros(self.init_shape, self.dtype))
