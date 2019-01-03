from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from zhusuan.distributions import Distribution
from zhusuan.distributions.utils import assert_same_float_dtype
from zhusuan.model.stochastic import StochasticTensor

import tensorflow as tf
import numpy as np


class MatrixVariateNormal(StochasticTensor):
    """
    The class of MatrixVariateNormal `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param mean: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, p). Each
        slice `[i, j, ..., k, :, :]` represents the mean matrix of the
        distribution.
    :param u: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` represents the row variance matrix of the
        distribution and should be positive definite.
    :param v: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]` represents the column variance matrix of the
        distribution and should be positive definite.
    :param u_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` uci has property uci uci^T = ui.
    :param v_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]`  vci has property vci vci^T = vi..
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """
    def __init__(self,
                 name,
                 mean,
                 u=None,
                 v=None,
                 u_c=None,
                 v_c=None,
                 u_c_logdet=None,
                 v_c_logdet=None,
                 n_samples=None,
                 group_event_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False):

        norm = DMatrixVariateNormal(
            mean,
            u=u,
            v=v,
            u_c=u_c,
            v_c=v_c,
            u_c_logdet=u_c_logdet,
            v_c_logdet=v_c_logdet,
            group_event_ndims=group_event_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics
        )
        super(MatrixVariateNormal, self).__init__(
            name, norm, n_samples)


class EigenMatrixVariateNormal(StochasticTensor):
    """
    The class of MatrixVariateNormal `StochasticTensor`.
    See :class:`~zhusuan.model.base.StochasticTensor` for details.

    :param mean: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, p). Each
        slice `[i, j, ..., k, :, :]` represents the mean matrix of the
        distribution.
    :param u: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` represents the row variance matrix of the
        distribution and should be positive definite.
    :param v: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]` represents the column variance matrix of the
        distribution and should be positive definite.
    :param u_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` uci has property uci uci^T = ui.
    :param v_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]`  vci has property vci vci^T = vi..
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """
    def __init__(self,
                 name,
                 mean,
                 u_b=None,
                 v_b=None,
                 r=None,
                 n_samples=None,
                 group_event_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False):

        norm = EigenMultivariateNormal(
            mean,
            u_b=u_b,
            v_b=v_b,
            r=r,
            group_event_ndims=group_event_ndims,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics
        )
        super(EigenMatrixVariateNormal, self).__init__(
            name, norm, n_samples)


class DMatrixVariateNormal(Distribution):
    """
    The class of Matrix variate Normal distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param mean: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, p). Each
        slice `[i, j, ..., k, :, :]` represents the mean matrix of the
        distribution.
    :param u: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` represents the row variance matrix of the
        distribution and should be positive definite.
    :param v: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]` represents the column variance matrix of the
        distribution and should be positive definite.
    :param u_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` uci has property uci uci^T = ui.
    :param v_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]`  vci has property vci vci^T = vi..
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 mean,
                 u=None,
                 v=None,
                 u_c=None,
                 v_c=None,
                 u_c_logdet = None,
                 v_c_logdet = None,
                 group_event_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False):

        mean = tf.convert_to_tensor(mean)
        _assert_rank_op = tf.assert_greater_equal(
            tf.rank(mean), 2,
            message="mean should be at least a 2-D tensor.")
        with tf.control_dependencies([_assert_rank_op]):
            self._mean = mean

        def _eig_decomp(mat):
            mat_t = transpose_last2dims(mat)
            e, v = tf.self_adjoint_eig((mat + mat_t) / 2 + tf.eye(tf.shape(mat)[-1]) * 1e-8)
            e = tf.maximum(e, 1e-10) ** 0.5
            return tf.matmul(v, tf.matrix_diag(e)), tf.reduce_sum(tf.log(e), -1)

        if u is not None and v is not None:
            # assert_same_rank([(self._mean, 'MatrixVariateNormal.mean'),
            #                   (u, 'MatrixVariateNormal.u'),
            #                   (v, 'MatrixVariateNormal.v')])
            u = tf.convert_to_tensor(u)
            _assert_shape_op_1 = tf.assert_equal(
                tf.shape(mean)[-2], tf.shape(u)[-1],
                message='second last dimension of mean should be the same \
                         as the last dimension of U matrix')
            _assert_shape_op_2 = tf.assert_equal(
                tf.shape(u)[-1], tf.shape(u)[-2],
                message='second last dimension of U should be the same \
                         as the last dimension of U matrix')
            with tf.control_dependencies([
                _assert_shape_op_1, _assert_shape_op_2,
                tf.check_numerics(u, 'U matrix')]):
                self._u = u
            v = tf.convert_to_tensor(v)
            _assert_shape_op_1 = tf.assert_equal(
                tf.shape(mean)[-1], tf.shape(v)[-1],
                message='last dimension of mean should be the same \
                         as last dimension of V matrix')
            _assert_shape_op_2 = tf.assert_equal(
                tf.shape(v)[-1], tf.shape(v)[-2],
                message='second last dimension of V should be the same \
                         as last dimension of V matrix')
            with tf.control_dependencies([
                _assert_shape_op_1, _assert_shape_op_2,
                tf.check_numerics(v, 'V matrix')]):
                self._v = v
            dtype = assert_same_float_dtype([(self._mean, 'MatrixVariateNormal.mean'),
                                             (self._u, 'MatrixVariateNormal.u'),
                                             (self._v, 'MatrixVariateNormal.v')])

            self._u_c, self._u_c_log_determinant = _eig_decomp(self._u)
            self._v_c, self._v_c_log_determinant = _eig_decomp(self._v)

        elif u_c is not None and v_c is not None:
            # assert_same_rank([(self._mean, 'MatrixVariateNormal.mean'),
            #                   (u_c, 'MatrixVariateNormal.u_c'),
            #                   (v_c, 'MatrixVariateNormal.v_c')])
            dtype = assert_same_float_dtype([(self._mean, 'MatrixVariateNormal.mean'),
                                             (u_c, 'MatrixVariateNormal.u_c'),
                                             (v_c, 'MatrixVariateNormal.v_c')])
            self._u_c = u_c
            self._v_c = v_c
            self._u = tf.matmul(self._u_c, transpose_last2dims(self._u_c))
            self._v = tf.matmul(self._v_c, transpose_last2dims(self._v_c))
            if u_c_logdet is not None:
                self._u_c_log_determinant = u_c_logdet
            else:
                _, self.u_c_log_determinant = _eig_decomp(self._u)
            if v_c_logdet is not None:
                self._v_c_log_determinant = v_c_logdet
            else:
                _, self._v_c_log_determinant = _eig_decomp(self._v)

        super(DMatrixVariateNormal, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_ndims=group_event_ndims)

    @property
    def mean(self):
        """The mean of the MatrixVariateNormal distribution."""
        return self._mean

    @property
    def u(self):
        """The row variance matrix of the MatrixVariateNormal distribution."""
        return self._u

    @property
    def v(self):
        """The column variance matrix of the MatrixVariateNormal distribution."""
        return self._v

    @property
    def u_c(self):
        """
        The cholesky decomposition of row variance matrix of the
        MatrixVariateNormal distribution.
        """
        return self._u_c

    @property
    def u_c_log_determinant(self):
        """
        The log determinant of the cholesky decomposition matrix of the row
        variance matrix.
        """
        return self._u_c_log_determinant

    @property
    def v_c(self):
        """
        The cholesky decomposition of column variance matrix of the
        MatrixVariateNormal distribution.
        """
        return self._v_c

    @property
    def v_c_log_determinant(self):
        """
        The log determinant of the cholesky decomposition matrix of the column
        variance matrix.
        """
        return self._v_c_log_determinant

    def _value_shape(self):
        return tf.shape(self.mean)[-2:]

    def _get_value_shape(self):
        return self.mean.get_shape()[-2:]

    def _batch_shape(self):
        return tf.shape(self.mean)[:-2]

    def _get_batch_shape(self):
        return self.mean.get_shape()[:-2]

    def _sample(self, n_samples):
        mean, u_c, v_c = self.mean, self.u_c, self.v_c
        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            u_c = tf.stop_gradient(u_c)
            v_c = tf.stop_gradient(v_c)
        u_c = tile_ntimes(u_c, n_samples)
        v_c = tile_ntimes(v_c, n_samples)

        shape = tf.concat([[n_samples], self.batch_shape, self.value_shape], 0)
        epsilon = tf.random_normal(shape, dtype=self.dtype)
        v_c_t = transpose_last2dims(v_c)

        samples = mean + tf.matmul(tf.matmul(u_c, epsilon), v_c_t)

        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()).concatenate(self.get_value_shape()))
        return samples

    def _log_prob(self, given):
        mean, u, v = self.mean, self.u, self.v
        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            u = tf.stop_gradient(u)
            v = tf.stop_gradient(v)
        u_inv = tile_ntimes(tf.matrix_inverse(u), tf.shape(given)[0])
        v_inv = tile_ntimes(tf.matrix_inverse(v), tf.shape(given)[0])
        E = given - mean
        Et = transpose_last2dims(given-mean)

        log_no = -0.5 * tf.trace(tf.matmul(tf.matmul(E, v_inv), tf.matmul(Et, u_inv)))
        p = tf.cast(tf.shape(mean)[-1], tf.float32)
        n = tf.cast(tf.shape(mean)[-2], tf.float32)
        log_de = 0.5 * n * p * np.log(2. * np.pi) \
            + n * self.v_c_log_determinant \
            + p * self.u_c_log_determinant
        log_prob = log_no - log_de
        return log_prob

    def _prob(self, given):
        return tf.exp(self._log_prob(self, given))


class EigenMultivariateNormal(Distribution):
    """
    The class of EigenMulti distribution.
    See :class:`~zhusuan.distributions.base.Distribution` for details.

    :param mean: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, p). Each
        slice `[i, j, ..., k, :, :]` represents the mean matrix of the
        distribution.
    :param u: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` represents the row variance matrix of the
        distribution and should be positive definite.
    :param v: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]` represents the column variance matrix of the
        distribution and should be positive definite.
    :param u_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., n, n). Each
        slice `[i, j, ..., k, :, :]` uci has property uci uci^T = ui.
    :param v_c: A (N+2)-D (N >= 0) `float` Tensor of shape (..., p, p). Each
        slice `[i, j, ..., k, :, :]`  vci has property vci vci^T = vi..
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 mean,
                 u_b=None,
                 v_b=None,
                 r=None,
                 group_event_ndims=0,
                 is_reparameterized=True,
                 check_numerics=False):

        mean = tf.convert_to_tensor(mean)
        _assert_rank_op = tf.assert_greater_equal(
            tf.rank(mean), 2,
            message="mean should be at least a 2-D tensor.")
        with tf.control_dependencies([_assert_rank_op]):
            self._mean = mean

        # assert_same_rank([(self._mean, 'EigenMatrixNormal.mean'),
        #                   (u_b, 'EigenMatrixNormal.u_b'),
        #                   (v_b, 'EigenMatrixNormal.v_b'),
        #                   (r, 'EigenMatrixNormal.r')])
        u_b = tf.convert_to_tensor(u_b)
        self._u_b = u_b

        # _assert_shape_op_1 = tf.assert_equal(
        #     tf.shape(mean)[-2], tf.shape(u)[-1],
        #     message='second last dimension of mean should be the same \
        #              as the last dimension of U matrix')
        # _assert_shape_op_2 = tf.assert_equal(
        #     tf.shape(u)[-1], tf.shape(u)[-2],
        #     message='second last dimension of U should be the same \
        #              as the last dimension of U matrix')
        # with tf.control_dependencies([
        #     _assert_shape_op_1, _assert_shape_op_2,
        #     tf.check_numerics(u, 'U matrix')]):

        v_b = tf.convert_to_tensor(v_b)
        self._v_b = v_b

        # _assert_shape_op_1 = tf.assert_equal(
        #     tf.shape(mean)[-1], tf.shape(v)[-1],
        #     message='last dimension of mean should be the same \
        #              as last dimension of V matrix')
        # _assert_shape_op_2 = tf.assert_equal(
        #     tf.shape(v)[-1], tf.shape(v)[-2],
        #     message='second last dimension of V should be the same \
        #              as last dimension of V matrix')
        # with tf.control_dependencies([
        #     _assert_shape_op_1, _assert_shape_op_2,
        #     tf.check_numerics(v, 'V matrix')]):

        r = tf.convert_to_tensor(r)
        self._r = r
        # _assert_shape_op_1 = tf.assert_equal(
        #     tf.shape(mean)[-1], tf.shape(r)[-1],
        #     message='second last dimension of mean should be the same \
        #                          as the last dimension of U matrix')
        # _assert_shape_op_2 = tf.assert_equal(
        #     tf.shape(mean)[-2], tf.shape(r)[-2],
        #     message='second last dimension of U should be the same \
        #                          as the last dimension of U matrix')
        # with tf.control_dependencies([
        #     _assert_shape_op_1, _assert_shape_op_2,
        #     tf.check_numerics(r, 'R matrix')]):
        #     self._r = r

        dtype = assert_same_float_dtype([(self._mean, 'MatrixVariateNormal.mean'),
                                         (self._u_b, 'MatrixVariateNormal.u_b'),
                                         (self._v_b, 'MatrixVariateNormal.v_b'),
                                         (self._r, 'MatrixVariateNormal.r')])

        # R should have been damped before. Sqrt for sampling.
        # self._r_c = tf.sqrt(self._r)
        self.log_std = 0.5 * tf.log(self._r)
        self.std = tf.exp(self.log_std)

        super(EigenMultivariateNormal, self).__init__(
            dtype=dtype,
            param_dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_ndims=group_event_ndims)

    @property
    def mean(self):
        """The mean of the MatrixVariateNormal distribution."""
        return self._mean

    @property
    def r(self):
        return self._r

    @property
    def u_b(self):
        return self._u_b

    @property
    def v_b(self):
        return self._v_b

    @property
    def r_c(self):
        return self._r_c

    def _value_shape(self):
        return tf.shape(self.mean)[-2:]

    def _get_value_shape(self):
        return self.mean.get_shape()[-2:]

    def _batch_shape(self):
        return tf.shape(self.mean)[:-2]

    def _get_batch_shape(self):
        return self.mean.get_shape()[:-2]

    def _sample(self, n_samples):
        mean, u_b, v_b, std = self.mean, self.u_b, self.v_b, self.std

        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            u_b = tf.stop_gradient(u_b)
            v_b = tf.stop_gradient(v_b)
            std = tf.stop_gradient(std)

        u_b = tile_ntimes(u_b, n_samples)
        v_b = tile_ntimes(v_b, n_samples)
        std = tile_ntimes(std, n_samples)

        shape = tf.concat([[n_samples], self.batch_shape, self.value_shape], 0)
        epsilon = tf.random_normal(shape, dtype=self.dtype)
        epsilon = tf.multiply(epsilon, std)

        v_b_t = transpose_last2dims(v_b)
        samples = mean + tf.matmul(u_b, tf.matmul(epsilon, v_b_t))

        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()).concatenate(self.get_value_shape()))
        return samples

    def _log_prob(self, given):
        raise NotImplementedError()

    def _prob(self, given):
        raise NotImplementedError()


def transpose_last2dims(mat):
    n = len(mat.get_shape())
    return tf.transpose(mat, list(range(n-2)) + [n-1, n-2])


def tile_ntimes(mat, n_particles):
    n = len(mat.get_shape())
    return tf.tile(tf.expand_dims(mat, 0), [n_particles] + [1]*n)
