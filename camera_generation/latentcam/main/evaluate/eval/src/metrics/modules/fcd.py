"""
Code highly adapted from:
https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
"""

from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics import Metric

# ------------------------------------------------------------------------------------- #


def _compute_fd(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)). # noqa

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)
    return a + b - 2 * c


class FrechetCLaTrDistance(Metric):
    """
    The Frechet Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is:
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2 * sqrt(sigm_1 *s igm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """

    def __init__(self, num_features: Union[int, Module] = 256, **kwargs):
        super().__init__(**kwargs)

        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, real_features, fake_features):
        """Update the state with extracted features."""
        
        # print(f"pred_feats device: {real_features.device}")
        # print(f"ref_feats device: {fake_features.device}")

        self.orig_dtype = real_features.dtype

        self.real_features_sum += real_features.sum(dim=0)
        self.real_features_cov_sum += real_features.t().mm(real_features)
        self.real_features_num_samples += real_features.shape[0]

        self.fake_features_sum += fake_features.sum(dim=0)
        self.fake_features_cov_sum += fake_features.t().mm(fake_features)
        self.fake_features_num_samples += fake_features.shape[0]

    def compute(self) -> Tensor:
        """
        Calculate FD_CLaTr score based on accumulated extracted features from the two
        distributions.
        """
        # 샘플이 1 개면 공분산 분모 (N-1) 이 0 이라 cov 가 통째로 NaN 이 되고, 그 NaN 행렬이
        # `torch.linalg.eigvals` 로 들어가면 **파이썬 예외가 아니라 MKL 이 죽는다**
        # ("Intel oneMKL ERROR: Parameter 3 was incorrect on entry to DGEBAL" -> rc=-11).
        # 프로세스가 통째로 날아가서 CLaTr/caption 결과도 같이 잃는다. FCD 는 두 분포를
        # 비교하는 지표라 N=1 에서는 정의 자체가 없으므로 NaN 으로 비우고 넘어간다.
        # ("영상 하나 + preset 하나" 요청(D219)이 항상 이 경로다. N>=2 는 기존과 동일.)
        n = int(min(self.real_features_num_samples, self.fake_features_num_samples))
        if n < 2:
            print(f"[fcd] 샘플이 {n} 개뿐이라 공분산을 못 만든다 — FCD 를 NaN 으로 둔다.")
            return torch.tensor(float("nan"), dtype=self.orig_dtype,
                                device=self.real_features_sum.device)

        mean_real = self.real_features_sum / self.real_features_num_samples
        mean_real = mean_real.unsqueeze(0)

        mean_fake = self.fake_features_sum / self.fake_features_num_samples
        mean_fake = mean_fake.unsqueeze(0)

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)

        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        fd = _compute_fd(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)

        return fd.to(self.orig_dtype)
