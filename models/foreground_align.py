import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class FC_AMMD(nn.Module):
    """
    Foreground Class-conditional Adaptive Multi-kernel Maximum Mean Discrepancy for support and query features.

    Args:
        n_way: number of foreground categories.
        max_per_class: number of downsample points per category.
        sigma_multipliers: kernel bandwidth.
        unbiased: unbiased estimation.
        num_neg: number of negative classes.
        gamma: margin for negative and positive classes.
        class_weight: strategy for the loss weights of classes.
        neg_strategy: negative classes selected strategy.
        hard_pool: Number of candidates in coarse screening
        median_sample_k: for the median estimation.
    """

    def __init__(
        self,
        n_way: int,
        max_per_class: int = 256,
        sigma_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        unbiased: bool = False,
        num_neg: int = 1,
        gamma: float = 0.1,
        detach_support_default: bool = True,
        class_weight: str = "sqrt_ns_nq",
        neg_strategy: str = "hard",
        hard_pool: int = 8,
        median_sample_k: int = 512,
        eps: float = 1e-12,
    ):
        super().__init__()
        assert n_way >= 1
        assert class_weight in ("sqrt_ns_nq", "sqrt_nq", "flat")
        assert neg_strategy in ("random", "hard")
        assert num_neg >= 0 and hard_pool >= 1
        self.n_way = n_way
        self.max_per_class = int(max_per_class)
        self._sigma_multipliers = tuple(float(s) for s in sigma_multipliers)
        self.unbiased = bool(unbiased)
        self.num_neg = int(num_neg)
        self.gamma = float(gamma)
        self.detach_support_default = bool(detach_support_default)
        self.class_weight = class_weight
        self.neg_strategy = neg_strategy
        self.hard_pool = int(hard_pool)
        self.median_sample_k = int(median_sample_k)
        self.eps = float(eps)

        self.register_buffer(
            "_sigma_grid", torch.tensor(self._sigma_multipliers, dtype=torch.float32),
            persistent=False
        )

    @staticmethod
    def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is None:
            y = x
        x2 = (x * x).sum(1, keepdim=True)
        y2 = (y * y).sum(1, keepdim=True).t()
        return (x2 + y2 - 2.0 * x @ y.t()).clamp_min(0.0)

    @staticmethod
    def _row_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + eps)

    @staticmethod
    def _subsample_rows(x: torch.Tensor, k: int) -> torch.Tensor:
        if k > 0 and x.shape[0] > k:
            idx = torch.randperm(x.shape[0], device=x.device)[:k]
            return x.index_select(0, idx)
        return x

    def _rbf_multi(self, sq_dists: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(sigmas):
            sigmas = torch.tensor(sigmas, device=sq_dists.device, dtype=sq_dists.dtype)
        sigmas = sigmas.view(-1, 1, 1)
        return torch.exp(-sq_dists.unsqueeze(0) / (2.0 * sigmas * sigmas)).sum(0)

    def _mmd2_from_sq(self, Da, Db, Dab, sigmas) -> torch.Tensor:
        Ka  = self._rbf_multi(Da,  sigmas)
        Kb  = self._rbf_multi(Db,  sigmas)
        Kab = self._rbf_multi(Dab, sigmas)
        na = Ka.shape[0]; nb = Kb.shape[0]
        if self.unbiased:
            Ka_sum = (Ka.sum() - Ka.diag().sum()) / max(na * (na - 1), 1)
            Kb_sum = (Kb.sum() - Kb.diag().sum()) / max(nb * (nb - 1), 1)
            mmd2 = Ka_sum + Kb_sum - 2.0 * Kab.mean()
        else:
            mmd2 = Ka.mean() + Kb.mean() - 2.0 * Kab.mean()
            mmd2 = torch.clamp(mmd2, min=0.0)
        return mmd2

    def _median_heuristic_sigmas_from_concat(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = torch.cat([xa, xb], 0)
            if z.shape[0] > self.median_sample_k:
                z = self._subsample_rows(z, self.median_sample_k)
            d = self._pairwise_sq_dists(z)
            mask = d > 0
            if mask.any():
                med = d[mask].median()
                base = torch.sqrt(med + 1e-6)
            else:
                base = torch.tensor(1.0, device=z.device, dtype=z.dtype)
            sigmas = base.to(z) * self._sigma_grid.to(z)
        return sigmas

    def forward(
        self,
        f_s: torch.Tensor,
        l_s: torch.Tensor,
        f_q: torch.Tensor,
        l_q: torch.Tensor,
        lambda_mmd: float = 0.1,
        detach_support: bool = False,
    ):
        if detach_support is None:
            detach_support = self.detach_support_default

        M_A, C_A, Na = f_s.shape
        M_B, C_B, Nb = f_q.shape
        assert C_A == C_B, "support/query with same dimensions"

        total = f_s.new_zeros(())
        valid_w = f_s.new_zeros(())

        f_q_mnc = f_q.permute(0, 2, 1).contiguous()

        support_bank: Dict[int, torch.Tensor] = {}
        ns_bank: Dict[int, int] = {}
        for c in range(1, self.n_way + 1):
            i = c - 1
            if i < 0 or i >= M_A:
                continue
            mask = (l_s[i] == 1)
            if mask.any():
                xa = f_s[i][:, mask].t()
                if xa.shape[0] >= 2:
                    support_bank[c] = xa
                    ns_bank[c] = xa.shape[0]

        query_bank: Dict[int, torch.Tensor] = {}
        nq_bank: Dict[int, int] = {}
        for c in range(1, self.n_way + 1):
            m_idx, n_idx = torch.where(l_q == c)
            if m_idx.numel() >= 2:
                xb = f_q_mnc[m_idx, n_idx]
                query_bank[c] = xb
                nq_bank[c] = xb.shape[0]

        margin_satisfied = []
        mmd_pos_list, mmd_neg_list = [], []

        for c in range(1, self.n_way + 1):
            if c not in support_bank or c not in query_bank:
                continue

            xa = self._row_norm(support_bank[c], self.eps)
            xb = self._row_norm(query_bank[c],   self.eps)
            xa = self._subsample_rows(xa, self.max_per_class)
            xb = self._subsample_rows(xb, self.max_per_class)
            if xa.shape[0] < 2 or xb.shape[0] < 2:
                continue
            if detach_support:
                xa = xa.detach()

            sig_pos = self._median_heuristic_sigmas_from_concat(xa, xb)
            Da  = self._pairwise_sq_dists(xa)
            Db  = self._pairwise_sq_dists(xb)
            Dab = self._pairwise_sq_dists(xa, xb)
            mmd_pos = self._mmd2_from_sq(Da, Db, Dab, sig_pos)
            mmd_pos_list.append(mmd_pos.detach())

            # select negative classes
            neg_classes_all = [k for k in support_bank.keys() if k != c]
            mmd_negs = []

            if self.num_neg > 0 and len(neg_classes_all) > 0:
                if self.neg_strategy == "random":
                    perm = torch.randperm(len(neg_classes_all), device=xa.device)
                    pick = [neg_classes_all[i.item()] for i in perm[: self.num_neg]]
                else:
                    # coarse stage
                    sig_screen = sig_pos
                    scores: List[Tuple[float, int]] = []
                    for k in neg_classes_all:
                        xa_k = self._row_norm(support_bank[k], self.eps)
                        xa_k = self._subsample_rows(xa_k, self.max_per_class)
                        if xa_k.shape[0] < 2:
                            continue

                        Da_k  = self._pairwise_sq_dists(xa_k)
                        Dab_k = self._pairwise_sq_dists(xa_k, xb)

                        mmd_scr = self._mmd2_from_sq(
                            Da_k, Db, Dab_k, sig_screen
                        ).detach()
                        scores.append((mmd_scr.item(), k))

                    # fine stage
                    if len(scores) > 0:
                        scores.sort(key=lambda t: t[0])
                        pool = [k for (_, k) in scores[: max(self.hard_pool, self.num_neg)]]
                        scored_exact = []
                        for k in pool:
                            xa_k = self._row_norm(support_bank[k], self.eps)
                            xa_k = self._subsample_rows(xa_k, self.max_per_class)
                            if xa_k.shape[0] < 2:
                                continue
                            if detach_support:
                                xa_k = xa_k.detach()
                            sig_k = self._median_heuristic_sigmas_from_concat(xa_k, xb)
                            Da_k  = self._pairwise_sq_dists(xa_k)
                            Dab_k = self._pairwise_sq_dists(xa_k, xb)
                            mmd_k = self._mmd2_from_sq(Da_k, Db, Dab_k, sig_k).detach()
                            scored_exact.append((mmd_k.item(), k))
                        scored_exact.sort(key=lambda t: t[0])
                        pick = [k for (_, k) in scored_exact[: self.num_neg]]
                    else:
                        pick = []

                for k in pick:
                    xa_k = self._row_norm(support_bank[k], self.eps)
                    xa_k = self._subsample_rows(xa_k, self.max_per_class)
                    if xa_k.shape[0] < 2:
                        continue
                    if detach_support:
                        xa_k = xa_k.detach()
                    sig_k = self._median_heuristic_sigmas_from_concat(xa_k, xb)
                    Da_k  = self._pairwise_sq_dists(xa_k)
                    Dab_k = self._pairwise_sq_dists(xa_k, xb)
                    mmd_k = self._mmd2_from_sq(Da_k, Db, Dab_k, sig_k)
                    mmd_negs.append(mmd_k)

            if len(mmd_negs) > 0:
                mmd_neg_mean = torch.stack(mmd_negs).mean()
                mmd_neg_list.append(mmd_neg_mean.detach())
                loss_c = torch.relu(mmd_pos - mmd_neg_mean + self.gamma)
                margin_satisfied.append(
                    (mmd_neg_mean - mmd_pos - self.gamma).detach()
                )
            else:
                loss_c = mmd_pos

            # class weight
            if self.class_weight == "sqrt_ns_nq":
                w = (ns_bank[c] * nq_bank[c]) ** 0.5
            elif self.class_weight == "sqrt_nq":
                w = (nq_bank[c]) ** 0.5
            else:
                w = 1.0
            w = torch.as_tensor(w, device=xa.device, dtype=xa.dtype)

            total = total + w * loss_c
            valid_w = valid_w + w

        if valid_w.item() == 0:
            return total

        loss = float(lambda_mmd) * (total / valid_w)

        # aux = {
        #     "valid_classes": int(valid_w.item() > 0),
        #     "avg_mmd_pos": (torch.stack(mmd_pos_list).mean().item() if len(mmd_pos_list) else 0.0),
        #     "avg_mmd_neg": (torch.stack(mmd_neg_list).mean().item() if len(mmd_neg_list) else 0.0),
        #     "margin_satisfied_ratio": (
        #         (torch.stack([m > 0 for m in margin_satisfied]).float().mean().item())
        #         if len(margin_satisfied) else 0.0
        #     ),
        # }
        # return loss, aux
        return loss
