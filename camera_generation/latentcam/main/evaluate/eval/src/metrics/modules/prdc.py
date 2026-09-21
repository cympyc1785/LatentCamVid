"""Code adapted from: https://github.com/clovaai/generative-evaluation-prdc"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import scipy

from utils.rotation_utils import pairwise_geodesic


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class ManifoldMetrics(Metric):
    def __init__(
        self,
        reset_real_features: bool = True,
        manifold_k: int = 3,
        distance: str = "geodesic",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.manifold_k = manifold_k
        self.reset_real_features = reset_real_features
        self.distance = distance

        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")

    # --------------------------------------------------------------------------------- #

    def _compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = torch.clone(data_x)

        if self.distance == "euclidean":
            num_feats = data_x.shape[-1]
            X = data_x.reshape(-1, num_feats).unsqueeze(0)
            Y = data_y.reshape(-1, num_feats).unsqueeze(0)
            dists = torch.cdist(X, Y, 2).squeeze(0)

        if self.distance == "geodesic":
            dists = pairwise_geodesic(data_x, data_y)

        return dists

    def _get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        # indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        # k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        # kth_values = k_smallests.max(axis=axis)

        k_smallests = torch.topk(unsorted, k, largest=False, dim=-1)
        kth_values = k_smallests.values.max(axis=axis).values
        return kth_values

    def _compute_nn_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def compute_prdc(self, real_features, fake_features, nearest_k):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        real_nearest_neighbour_distances = self._compute_nn_distances(
            real_features, nearest_k
        )
        fake_nearest_neighbour_distances = self._compute_nn_distances(
            fake_features, nearest_k
        )
        distance_real_fake = self._compute_pairwise_distance(
            real_features, fake_features
        )

        precision = (
            (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1))
            .any(axis=0)
            .to(float)
        ).mean()

        recall = (
            (distance_real_fake < fake_nearest_neighbour_distances.unsqueeze(1))
            .any(axis=1)
            .to(float)
            .mean()
        )

        # print(distance_real_fake)
        # print(real_nearest_neighbour_distances.unsqueeze(1))
        # print(distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1))
        # # print(nearest_k)
        # # print((
        # #     distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
        # # ).sum(axis=0).to(float).mean())
        # # print((distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)).shape)
        # print((
        #     distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
        # ).sum(axis=0))
        density = (1.0 / float(nearest_k)) * (
            distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
        ).sum(axis=0).to(float).mean()

        coverage = (
            (distance_real_fake.min(axis=1).values < real_nearest_neighbour_distances)
            .to(float)
            .mean()
        )
        return precision, recall, density, coverage

    # --------------------------------------------------------------------------------- #

    def update(self, real_features, fake_features):
        """Updates the state with new real and fake features."""
        self.real_features.append(real_features)
        self.fake_features.append(fake_features)

    def compute(self, num_splits=5):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            nearest_k: int.
            num_splits: int. Number of splits to use for computing metrics.
        Returns:
            dict of precision, recall, density, and coverage.
        """

        real_all = dim_zero_cat(self.real_features)
        fake_all = dim_zero_cat(self.fake_features)

        # split 하나가 manifold_k+1 개보다 작으면 topk 가 "selected index k out of range" 로
        # 죽고 metrics.json 자체가 안 나온다 (FCD/caption 까지 같이 날아간다). val 이 작은
        # 코퍼스(TRUMANS-Lite: val 14 세그먼트 -> chunk(5) = 3,3,3,3,2 < k+1=4)에서만 걸린다.
        # N >= num_splits*(manifold_k+1) 이면 아래 계산이 num_splits 를 그대로 써서 기존 arm 과
        # 완전히 동일하다 (DL3DV val 3263 -> 5).
        n = min(real_all.shape[0], fake_all.shape[0])
        need = self.manifold_k + 1
        eff_splits = max(1, min(num_splits, n // need))
        if eff_splits != num_splits:
            print(
                f"[prdc] N={n} 이라 num_splits {num_splits} -> {eff_splits} "
                f"(split 당 최소 {need} 개 필요)"
            )
        if n < need:
            # 예전엔 여기서 raise 했는데, 그러면 metrics.json 자체가 안 나와서 **CLaTr/FCD/caption
            # 까지 같이 날아간다**. "영상 하나 + preset 하나" 요청(D219)은 test 가 1 세그먼트라
            # 항상 이 경로를 타고, 정작 보려는 건 PRDC 가 아니라 clatr_score/caption fscore 다.
            # PRDC 만 NaN 으로 비우고 나머지 지표는 살린다 — N>=need 인 기존 arm 은 영향 없다.
            print(
                f"[prdc] 샘플이 {n} 개뿐이라 manifold_k={self.manifold_k} 로는 PRDC 를 못 잰다 "
                f"(최소 {need} 개) — PRDC 4 개 지표만 NaN 으로 두고 나머지는 계속 잰다."
            )
            nan = torch.tensor(float("nan"), device=real_all.device)
            return nan, nan.clone(), nan.clone(), nan.clone()

        real_features = real_all.chunk(eff_splits, dim=0)
        fake_features = fake_all.chunk(eff_splits, dim=0)
        precision, recall, density, coverage = [], [], [], []
        for real, fake in zip(real_features, fake_features):
            # torch.chunk 는 마지막 조각이 짧을 수 있다 (N=17, splits=4 -> 5,5,5,2).
            # 짧은 꼬리는 버린다 — 앞 조각들이 이미 전체를 대표한다.
            if min(real.shape[0], fake.shape[0]) < need:
                continue
            p, r, d, c = self.compute_prdc(real, fake, nearest_k=self.manifold_k)
            precision.append(p)
            recall.append(r)
            density.append(d)
            coverage.append(c)

        precision = torch.stack(precision).mean()
        recall = torch.stack(recall).mean()
        density = torch.stack(density).mean()
        coverage = torch.stack(coverage).mean()

        return precision, recall, density, coverage
