from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Mapping, Optional
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import math
import warnings


def select_clusters(
    df_datos: pd.DataFrame,
    df_reglas: pd.DataFrame,
    keep_all_clusters: bool = True,
    fallback_cluster: float | None = None,
):
    """Determine cluster assignments for each record based on rules.

    Parameters
    ----------
    df_datos : pd.DataFrame
        DataFrame containing the data to assign to clusters.
    df_reglas : pd.DataFrame
        DataFrame defining the rules with MultiIndex columns where:
          - Lower limits are under first level 'linf'
          - Upper limits are under first level 'lsup'
          - Weight is stored at ('metrics', 'ponderador')
          - The 'cluster' column appears normally.
    keep_all_clusters : bool, optional
        If True, also store all clusters and their weights satisfied by each
        record. Otherwise, only the main cluster (highest weight) is kept.
    fallback_cluster : float, optional
        Cluster to assign to records that do not match any rule. If ``None``
        (default), unassigned records remain with value ``-1`` and a warning is
        issued.

    Returns
    -------
    clusters_datos : np.ndarray
        Array with the selected cluster for each record (-1 if none).
    clusters_datos_all : list[list[float]] | None
        For each record, list of clusters it belongs to (only if
        ``keep_all_clusters``).
    ponderadores_datos_all : list[list[float]] | None
        For each record, list of weights corresponding to
        ``clusters_datos_all`` (only if ``keep_all_clusters``).
    """

    n_datos = df_datos.shape[0]
    clusters_datos = np.full(n_datos, -1, dtype=float)
    ponderador_datos = np.full(n_datos, -np.inf, dtype=float)
    clusters_datos_all = [[] for _ in range(n_datos)] if keep_all_clusters else None
    ponderadores_datos_all = [[] for _ in range(n_datos)] if keep_all_clusters else None

    col_to_idx = {col: idx for idx, col in enumerate(df_datos.columns)}
    X_values = df_datos.to_numpy()

    reglas_info = []
    for _, row in df_reglas.iterrows():
        if row[('metrics', 'ponderador')] == 0:
            continue
        linf = row['linf'].dropna()
        lsup = row['lsup'].dropna()
        variables = linf.index.tolist()

        try:
            idx = np.array([col_to_idx[var] for var in variables], dtype=int)
        except KeyError as err:
            missing_cols = [var for var in variables if var not in col_to_idx]
            raise KeyError(f"Columns not found in df_datos: {missing_cols}") from err

        linf_vals = (
            np.asarray([linf[var] for var in variables], dtype=float)
            if variables
            else np.array([], dtype=float)
        )
        lsup_vals = (
            np.asarray([lsup[var] for var in variables], dtype=float)
            if variables
            else np.array([], dtype=float)
        )

        p_val = row[('metrics', 'ponderador')]
        ponderador = p_val.mean() if hasattr(p_val, '__iter__') else p_val

        cluster_raw = row['cluster']
        if hasattr(cluster_raw, 'values') and len(cluster_raw.values) == 1:
            cluster_raw = float(cluster_raw.values[0])
        else:
            cluster_raw = float(cluster_raw)

        reglas_info.append(
            {
                'variables': variables,
                'idx': idx,
                'linf': linf_vals,
                'lsup': lsup_vals,
                'ponderador': ponderador,
                'cluster': cluster_raw,
            }
        )

    for regla in reglas_info:
        idx = regla['idx']
        linf_vals = regla['linf']
        lsup_vals = regla['lsup']
        ponderador = regla['ponderador']
        cluster = regla['cluster']

        if idx.size:
            X_sub = X_values[:, idx]
            condiciones = np.logical_and(X_sub >= linf_vals, X_sub <= lsup_vals)
            cumple_regla = condiciones.all(axis=1)
        else:
            cumple_regla = np.zeros(n_datos, dtype=bool)

        if keep_all_clusters:
            indices_cumple = np.where(cumple_regla)[0]
            for i in indices_cumple:
                clusters_datos_all[i].append(cluster)
                ponderadores_datos_all[i].append(ponderador)

        actualizar = cumple_regla & (ponderador > ponderador_datos)
        clusters_datos[actualizar] = cluster
        ponderador_datos[actualizar] = ponderador

    # Detect records without assigned cluster after evaluating all rules
    indices_sin_cluster = np.where(clusters_datos == -1)[0]
    if len(indices_sin_cluster) > 0:
        if fallback_cluster is not None:
            clusters_datos[indices_sin_cluster] = fallback_cluster
            if keep_all_clusters:
                for i in indices_sin_cluster:
                    clusters_datos_all[i].append(fallback_cluster)
                    ponderadores_datos_all[i].append(0.0)
        else:
            warnings.warn(
                f"{len(indices_sin_cluster)} records did not match any rule.",
                UserWarning,
            )

    return clusters_datos, clusters_datos_all, ponderadores_datos_all


class MenuClusterSelector:
    """
    Cluster selector when X = records (one menu per row).
    Trains with y to estimate q_v(y). During prediction chooses a value per row
    maximizing a global objective: J = w_nmi * NMI + w_v * V_measure - λ * RegK.

    - fit(records_train, y): compute q_v(y) (Laplace smoothing) and set the
      vocabulary of values.
    - predict(records, n_clusters=None): assign one value per row without seeing y,
      optimizing J via coordinate ascent (greedy improvements). If n_clusters=K,
      first restrict to a catalog S of size K using coverage+quality and then optimize.
    """

    # =========================
    #   METRICS (explicit)
    # =========================
    @staticmethod
    def _safe_div(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(b > 0, a / b, 0.0)
        return r

    @classmethod
    def _nmi_from_soft(cls, C: np.ndarray, Py: np.ndarray, Pv: np.ndarray,
                       average: str = "arithmetic") -> float:
        """
        NMI sobre 'contingencia suave': C[t,k] suma de q_v(y=t) de filas asignadas al valor k.
        Py[t] = #filas clase t, Pv[k] = #filas asignadas al valor k.
        """
        n = C.sum()
        if n <= 0:
            return 0.0
        Ptk = C / n
        Pt = Py / n
        Pk = Pv / n

        denom = np.clip(Pt[:, None] * Pk[None, :], 1e-12, None)
        ratio = np.clip(Ptk / denom, 1e-12, None)
        MI = float((Ptk * np.log(ratio)).sum())

        Hy = float(-(Pt * np.log(np.clip(Pt, 1e-12, 1.0))).sum())
        Hv = float(-(Pk * np.log(np.clip(Pk, 1e-12, 1.0))).sum())
        if Hy == 0.0 or Hv == 0.0:
            return 0.0

        if average == "arithmetic":
            denomN = (Hy + Hv) / 2.0
        elif average == "geometric":
            denomN = (Hy * Hv) ** 0.5
        else:  # "min"
            denomN = min(Hy, Hv)
        return float(MI / max(denomN, 1e-12))

    @classmethod
    def _v_measure_from_soft(cls, C: np.ndarray, Py: np.ndarray, Pv: np.ndarray,
                             beta: float = 1.0) -> float:
        """
        V-measure = harmonic(homogeneity, completeness) over C, Py, Pv.
        """
        n = C.sum()
        if n <= 0:
            return 0.0
        Ptk = C / n
        Pt = Py / n
        Pk = Pv / n

        # Homogeneidad: 1 - H(Y|V)/H(Y)
        Py_k = cls._safe_div(Ptk, Pk[None, :])
        Py_k = np.clip(Py_k, 1e-12, 1.0)
        Hy_given_V = float(-(Ptk * np.log(Py_k)).sum())
        Hy = float(-(Pt * np.log(np.clip(Pt, 1e-12, 1.0))).sum())
        hom = 0.0 if Hy == 0.0 else 1.0 - Hy_given_V / Hy

        # Completitud: 1 - H(V|Y)/H(V)
        Pv_t = cls._safe_div(Ptk, Pt[:, None])
        Pv_t = np.clip(Pv_t, 1e-12, 1.0)
        Hv_given_Y = float(-(Ptk * np.log(Pv_t)).sum())
        Hv = float(-(Pk * np.log(np.clip(Pk, 1e-12, 1.0))).sum())
        comp = 0.0 if Hv == 0.0 else 1.0 - Hv_given_Y / Hv

        if hom == 0.0 and comp == 0.0:
            return 0.0
        return float((1 + beta) * hom * comp / (beta * hom + comp + 1e-12))

    @staticmethod
    def _k_regularizer(Pv: np.ndarray, target_K: int | None, lam: float) -> float:
        """
        Global regularizer over the number of values used:
          - If target_K is None: Reg = lam * H(V) (penalizes high entropy ⇒ fewer effective values).
          - If target_K is int:  Reg = lam * (H(V) - log(target_K))^2 (pushes toward ~K values).
        """
        if lam <= 0.0:
            return 0.0
        n = Pv.sum()
        if n <= 0:
            return 0.0
        Pk = np.clip(Pv / n, 1e-12, 1.0)
        Hv = float(-(Pk * np.log(Pk)).sum())
        if target_K is None:
            return lam * Hv
        else:
            return lam * (Hv - np.log(max(int(target_K), 1)))**2

    # ======================
    #      INTERFAZ
    # ======================
    def __init__(
        self,
        w_nmi: float = 1.0,
        w_v: float = 1.0,
        lam_k: float = 0.1,
        target_K: int | None = None,
        smoothing: float = 1.0,
        max_passes: int = 5,
        tol: float = 1e-6,
        seed: int | None = 42,
    ):
        self.w_nmi = w_nmi
        self.w_v = w_v
        self.lam_k = lam_k
        self.target_K = target_K
        self.smoothing = smoothing
        self.max_passes = max_passes
        self.tol = tol
        self.seed = seed

        # Se rellenan en fit()
        self.classes_: np.ndarray | None = None      # orden de clases
        self.value_to_idx_: Dict[Any, int] = {}
        self.idx_to_value_: List[Any] = []
        self.q_: np.ndarray | None = None            # (V, T) q_v(y)
        self.Py_: np.ndarray | None = None           # (T,) conteo por clase en train

    # ---------- helpers de vocabulario ----------
    def _build_vocab(self, records: Sequence[Sequence[Any]]):
        Vset = set()
        for row in records:
            if not row:
                Vset.add(None)
            else:
                Vset.update(row)
        self.idx_to_value_ = sorted(Vset, key=lambda x: (x is None, str(x)))
        self.value_to_idx_ = {v: i for i, v in enumerate(self.idx_to_value_)}

    def _ensure_vocab_for_predict(self, records: Sequence[Sequence[Any]]):
        # Add unseen values from train with uniform q_v (smoothing)
        new_vals = []
        for row in records:
            for v in (row if row else [None]):
                if v not in self.value_to_idx_:
                    new_vals.append(v)
        if not new_vals:
            return
        # ampliar estructuras
        start = len(self.idx_to_value_)
        for j, v in enumerate(new_vals, start=start):
            self.value_to_idx_[v] = j
            self.idx_to_value_.append(v)
        V_new = len(self.idx_to_value_)
        T = len(self.classes_)
        q_new = np.full((V_new, T), 1.0 / T, dtype=np.float64)
        q_new[: self.q_.shape[0], :] = self.q_
        self.q_ = q_new  # unseen -> uniforme

    def _menus_indices(self, records: Sequence[Sequence[Any]]) -> List[np.ndarray]:
        idxs = []
        for row in records:
            row = row if row else [None]
            idxs.append(np.array([self.value_to_idx_[v] for v in row], dtype=int))
        return idxs

    # --------------- FIT ----------------
    def fit(self, records: Sequence[Sequence[Any]], y: Sequence[Any]):
        """
        Estima q_v(y) con suavizado Laplace sobre TODAS las filas donde v estuvo disponible.
        """
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        T = len(self.classes_)
        self.Py_ = np.bincount(y_idx, minlength=T).astype(np.float64)

        self._build_vocab(records)
        V = len(self.idx_to_value_)

        counts = np.zeros((V, T), dtype=np.float64)
        for i, row in enumerate(records):
            opts = row if row else [None]
            for v in opts:
                counts[self.value_to_idx_[v], y_idx[i]] += 1.0

        counts += float(self.smoothing)  # Laplace
        counts /= counts.sum(axis=1, keepdims=True)  # q_v(y) por fila de v
        self.q_ = counts  # (V, T)
        return self

    # --------------- PREDICT ----------------
    def _objective(self, C: np.ndarray, Py: np.ndarray, Pv: np.ndarray) -> float:
        """Compute the global objective J for a given contingency."""
        nmi = self._nmi_from_soft(C, Py, Pv)
        vms = self._v_measure_from_soft(C, Py, Pv)
        reg = self._k_regularizer(Pv, self.target_K, self.lam_k)
        return self.w_nmi * nmi + self.w_v * vms - reg

    def _build_catalog(
        self,
        allowed: List[np.ndarray],
        n_clusters: int | None,
        V: int,
        Py: np.ndarray,
    ) -> List[np.ndarray]:
        """Restrict menus to a catalog ``S`` of size ``n_clusters`` if provided."""
        if n_clusters is None:
            return allowed

        n = len(allowed)
        remaining = set(range(n))
        S: set[int] = set()
        Pt = Py / Py.sum()
        s_val = np.log(np.clip(self.q_, 1e-12, 1.0)) @ Pt  # (V,)

        while remaining:
            cover_gain = []
            for v in range(V):
                gain = sum(s_val[v] for i in remaining if v in allowed[i])
                cover_gain.append((gain, v))
            v_best = max(cover_gain)[1]
            S.add(v_best)
            covered = [i for i in list(remaining) if v_best in allowed[i]]
            for i in covered:
                remaining.discard(i)

        k_min = len(S)
        K = max(k_min, int(n_clusters))
        if K > k_min:
            extras = sorted(
                (v for v in range(V) if v not in S),
                key=lambda v: s_val[v],
                reverse=True,
            )[: K - k_min]
            S.update(extras)
        elif K < k_min:
            pass  # imposible cubrir con menos de k_min; usamos k_min

        S_arr = np.array(sorted(S), dtype=int)
        for i in range(n):
            inter = np.intersect1d(allowed[i], S_arr, assume_unique=False)
            allowed[i] = inter if inter.size > 0 else allowed[i]
        return allowed

    def _initial_assignment(
        self, allowed: List[np.ndarray], Py: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return initial assignment and contingency matrices."""
        Pt = Py / Py.sum()
        base_score_v = np.log(np.clip(self.q_, 1e-12, 1.0)) @ Pt
        n = len(allowed)
        V, T = self.q_.shape
        assign = np.empty(n, dtype=int)
        for i in range(n):
            cand = allowed[i]
            assign[i] = cand[np.argmax(base_score_v[cand])]

        C = np.zeros((T, V), dtype=np.float64)
        Pv = np.zeros(V, dtype=np.float64)
        for i in range(n):
            v = assign[i]
            C[:, v] += self.q_[v]
            Pv[v] += 1.0

        curJ = self._objective(C, Py, Pv)
        return assign, C, Pv, curJ

    def _coordinate_ascent(
        self,
        allowed: List[np.ndarray],
        assign: np.ndarray,
        C: np.ndarray,
        Pv: np.ndarray,
        Py: np.ndarray,
        curJ: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Optimize assignments via coordinate ascent."""
        n = len(assign)
        improved = True
        passes = 0
        while improved and passes < self.max_passes:
            improved = False
            passes += 1
            order = rng.permutation(n)
            for i in order:
                v_cur = assign[i]
                best_v = v_cur
                best_J = curJ

                C[:, v_cur] -= self.q_[v_cur]
                Pv[v_cur] -= 1.0

                for v_new in allowed[i]:
                    C[:, v_new] += self.q_[v_new]
                    Pv[v_new] += 1.0

                    J_new = self._objective(C, Py, Pv)
                    if J_new > best_J + self.tol:
                        best_J = J_new
                        best_v = v_new

                    C[:, v_new] -= self.q_[v_new]
                    Pv[v_new] -= 1.0

                C[:, best_v] += self.q_[best_v]
                Pv[best_v] += 1.0
                assign[i] = best_v
                if best_v != v_cur:
                    curJ = best_J
                    improved = True
        return assign

    def predict(self, records: Sequence[Sequence[Any]], n_clusters: int | None = None) -> List[Any]:
        """
        Assign one value per row maximizing J = w_nmi*NMI + w_v*V - lam_k*RegK,
        using coordinate ascent (greedy improvements).
        If n_clusters=K, first restrict to a catalog S of size K (coverage+quality).
        """
        assert (
            self.q_ is not None and self.classes_ is not None and self.Py_ is not None
        ), "Llama fit() primero."
        rng = np.random.default_rng(self.seed)

        self._ensure_vocab_for_predict(records)
        V, _ = self.q_.shape
        Py = self.Py_.copy()
        menus_idx = self._menus_indices(records)
        allowed = [np.array(opts, dtype=int) for opts in menus_idx]

        allowed = self._build_catalog(allowed, n_clusters, V, Py)
        assign, C, Pv, curJ = self._initial_assignment(allowed, Py)
        assign = self._coordinate_ascent(allowed, assign, C, Pv, Py, curJ, rng)

        return [self.idx_to_value_[j] for j in assign]


def balance_lists_n_clusters(
    records: Sequence[Sequence[Any]],
    n_clusters: int | None = None,
    *,
    max_iter: int = 20_000,
    restarts: int = 4,
    T0: float = 1.0,
    alpha: float = 0.999,
    seed: int | None = None,
) -> List[Any]:
    """
    Assign **a single value per row** optimizing two objectives with identical weight:

    • |distinct - n_clusters| → approach the desired number of clusters
      (if `n_clusters` is `None`, the minimum possible is chosen naturally).

    • Absolute imbalance → Σ |c_v – ideal| / n, where ``ideal = n / k``.
    """
    rng = random.Random(seed)
    records = [row if row else [-1] for row in records]
    n = len(records)

    # Utilidades internas -------------------------------------------------
    def imbalance(cnt: Counter):
        k = len(cnt)
        if k == 0:
            return 1.0
        ideal = n / k
        return sum(abs(c - ideal) for c in cnt.values()) / n

    def score(assign: List[Any]) -> float:
        cnt = Counter(assign)
        k = len(cnt)
        if n_clusters is None:
            cluster_term = k / n  # minimizar k
        else:
            cluster_term = abs(k - n_clusters) / n
        return cluster_term + imbalance(cnt)

    def neighbour(assign: List[Any]) -> List[Any]:
        """Move one row to another valid option (random)."""
        i = rng.randrange(n)
        row = records[i]
        cur = assign[i]
        alt = [v for v in row if v != cur]
        if not alt:  # fila sin alternativas
            return assign
        new = assign[:]
        new[i] = rng.choice(alt)
        return new

    # Reasonable initialization ------------------------------------------
    val_rows = defaultdict(list)
    for idx, row in enumerate(records):
        for v in row:
            val_rows[v].append(idx)

    remaining = set(range(n))
    chosen: List[Any] = []
    while remaining:
        best = max(val_rows, key=lambda v: len(set(val_rows[v]) & remaining))
        chosen.append(best)
        remaining -= set(val_rows[best])

    if n_clusters is not None and len(chosen) < n_clusters:
        extras = sorted(
            (v for v in val_rows if v not in chosen),
            key=lambda v: -len(val_rows[v]),
        )
        chosen.extend(extras[: n_clusters - len(chosen)])

    def initial_assignment() -> List[Any]:
        cnt: Counter = Counter()
        assign: List[Any] = [None] * n
        for i, row in enumerate(records):
            opts = [v for v in row if v in chosen] or row
            v = min(opts, key=lambda x: (cnt[x], x))
            assign[i] = v
            cnt[v] += 1
        return assign

    # Simulated Annealing -------------------------------------------------
    best_global, best_score = None, float("inf")
    for _ in range(restarts):
        cur = initial_assignment()
        cur_score = score(cur)
        best_local, best_local_score = cur[:], cur_score
        T = T0
        for _ in range(max_iter):
            nxt = neighbour(cur)
            if nxt is cur:
                continue
            nxt_score = score(nxt)
            accept = nxt_score < cur_score or rng.random() < math.exp((cur_score - nxt_score) / T)
            if accept:
                cur, cur_score = nxt, nxt_score
                if cur_score < best_local_score:
                    best_local, best_local_score = cur[:], cur_score
            T *= alpha
        if best_local_score < best_score:
            best_global, best_score = best_local, best_local_score

    return best_global


def max_prob_clusters(
    records: Sequence[Sequence[Any]],
    probs: Mapping[Any, float],
    n_clusters: int | None = None,
    *,
    max_iter: int = 20_000,
    restarts: int = 4,
    T0: float = 1.0,
    alpha: float = 0.999,
    seed: int | None = None,
) -> List[Any]:
    """
    Select **one value per row** such that:
      • If `n_clusters` is `None` → minimize number of distinct values.
      • If `n_clusters` is an integer:
          – attempt to return EXACTLY that number of clusters, maximizing the sum of probabilities.
          – if impossible, use the nearest feasible value (`k_min` or `k_max`).
    """
    rng = random.Random(seed)
    n = len(records)
    records = [row if row else [None] for row in records]

    # Paso 1: greedy set-cover para k_min -------------------------------
    value_rows = defaultdict(set)
    for i, row in enumerate(records):
        for v in row:
            value_rows[v].add(i)

    remaining = set(range(n))
    S: set[Any] = set()
    while remaining:
        best = max(
            value_rows,
            key=lambda v: (len(value_rows[v] & remaining), probs.get(v, 0.0)),
        )
        S.add(best)
        remaining -= value_rows[best]

    k_min = len(S)
    k_max = len(value_rows)

    # Paso 2: determinar k_target ---------------------------------------
    if n_clusters is None:
        k_target = k_min
    else:
        k_target = max(k_min, min(n_clusters, k_max))

    if k_target > k_min:
        extras = sorted(
            (v for v in value_rows if v not in S),
            key=lambda v: probs.get(v, 0.0),
            reverse=True,
        )
        S.update(extras[: k_target - k_min])

    S = set(list(S)[:k_target])  # asegura |S| == k_target

    # Step 3: greedy assignment -----------------------------------------
    assign: List[Any] = []
    for row in records:
        opts = [v for v in row if v in S]
        if not opts:
            best = max(row, key=lambda v: probs.get(v, 0.0))
            if best not in S and len(S) == k_target:
                worst = min(S, key=lambda v: probs.get(v, 0.0))
                S.remove(worst)
                S.add(best)
            opts = [v for v in row if v in S]
        assign.append(max(opts, key=lambda v: probs.get(v, 0.0)))

    # Paso 4: Simulated Annealing ---------------------------------------
    B = n + 1  # peso que penaliza cambiar k

    def cost(ass: List[Any]) -> float:
        k = len(set(ass))
        return abs(k - k_target) * B - sum(probs.get(v, 0.0) for v in ass)

    def neighbour(ass: List[Any]) -> List[Any]:
        i = rng.randrange(n)
        cur_row = records[i]
        cur_v = ass[i]
        alt = [v for v in cur_row if v in S and v != cur_v]
        if not alt:
            return ass
        new = ass[:]
        new[i] = rng.choice(alt)
        return new

    best_global, best_c = assign[:], cost(assign)
    for _ in range(restarts):
        cur, cur_c = assign[:], best_c
        T = T0
        for _ in range(max_iter):
            nxt = neighbour(cur)
            if nxt is cur:
                T *= alpha
                continue
            nxt_c = cost(nxt)
            if nxt_c < cur_c or rng.random() < math.exp((cur_c - nxt_c) / T):
                cur, cur_c = nxt, nxt_c
                if cur_c < best_c:
                    best_global, best_c = cur[:], cur_c
            T *= alpha

    return best_global


def match_class_distribution(
    records: Sequence[Sequence[Any]],
    y: Sequence[Any],
    n_clusters: int | None = None,
    *,
    seed: int | None = None,
) -> List[Any]:
    """Assign a value per row imitating the distribution of ``y``.

    Parameters
    ----------
    records : Sequence[Sequence[Any]]
        List of label options per row.
    y : Sequence[Any]
        Target classes associated with each row.
    n_clusters : int | None, optional
        Desired number of distinct labels. Used as a soft bound,
        prioritizing the most frequent.
    seed : int | None, optional
        Random seed for processing order.

    Returns
    -------
    List[Any]
        Label selected for each row.
    """

    rng = np.random.default_rng(seed)
    n = len(records)
    records = [row if row else [None] for row in records]

    y = np.asarray(y)
    clases, y_idx = np.unique(y, return_inverse=True)
    T = len(clases)
    Py = np.bincount(y_idx, minlength=T).astype(float)
    Py /= Py.sum()

    # Limitar a n_clusters valores si se solicita (cota blanda)
    freq = Counter(v for row in records for v in row)
    if n_clusters is not None and len(freq) > n_clusters:
        allowed_vals = set(v for v, _ in freq.most_common(n_clusters))
    else:
        allowed_vals = set(freq)

    # Conteos por etiqueta y clase
    cluster_counts: Dict[Any, np.ndarray] = {}

    orden = rng.permutation(n)
    asignacion: List[Any] = [None] * n
    for i in orden:
        opts = [v for v in records[i] if v in allowed_vals] or records[i]
        mejor_v = None
        mejor_score = float("inf")
        for v in opts:
            counts = cluster_counts.get(v)
            if counts is None:
                counts_tmp = np.zeros(T, dtype=float)
            else:
                counts_tmp = counts.copy()
            counts_tmp[y_idx[i]] += 1.0
            P = counts_tmp / counts_tmp.sum()
            score = np.abs(P - Py).sum()
            if score < mejor_score:
                mejor_score = score
                mejor_v = v
        asignacion[i] = mejor_v
        counts = cluster_counts.setdefault(mejor_v, np.zeros(T, dtype=float))
        counts[y_idx[i]] += 1.0

    return asignacion


# -------------------------------
# Utilidades de probabilidad
# -------------------------------


def _as_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, None)
    s = float(x.sum())
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def _cv(p: np.ndarray) -> float:
    m = float(p.mean())
    if m == 0:
        return 0.0
    return float(p.std() / (m + 1e-12))


def _round_quota(pi: np.ndarray, n: int) -> np.ndarray:
    raw = pi * n
    flo = np.floor(raw).astype(int)
    rem = int(n - flo.sum())
    if rem > 0:
        order = np.argsort(-(raw - flo))
        flo[order[:rem]] += 1
    return flo


def compress_distribution_to_K(Py: np.ndarray, K: int) -> np.ndarray:
    """
    Compress the distribution of y (counts Py) to K masses without losing "shape"
    by repeatedly merging the two smallest masses.
    Returns proportions that sum to 1.
    """
    masses = list(np.asarray(Py, dtype=np.float64))
    if K >= len(masses):
        return _as_prob(np.asarray(masses, dtype=np.float64))
    masses.sort()
    while len(masses) > K:
        a = masses.pop(0)
        b = masses.pop(0)
        masses.append(a + b)
        masses.sort()
    masses.sort(reverse=True)
    return _as_prob(np.asarray(masses, dtype=np.float64))


# ---------------------------------------
# Selector: etiquetas = valores de records
# ---------------------------------------


class ChimeraValuesSelector:
    """
    Assign ONE value per row (always from its own menu) such that:
      - The number of DISTINCT values (K) can be fixed or auto-chosen.
      - The frequency distribution per chosen value mimics the shape of y
        (compress Py→K masses and translate those quotas to K actual values).
      - The semantic quality of value v is measured with s(v) = log q_v · P(y).

    Flow:
      fit(records_train, y_train):
        - Learn q_v(y) with Laplace smoothing over availability.
      predict(records, n_labels=None, k_range=(2,12)):
        - Build catalog S of K values (set-cover + quality).
        - Assign target quotas ~ pi_K*n to each value of S (respecting capacities).
        - Assign each row to its best option in S with available capacity.
    """

    def __init__(self, smoothing: float = 1.0, seed: Optional[int] = 42):
        self.smoothing = smoothing
        self.seed = seed

        self.classes_: Optional[np.ndarray] = None
        self.Py_: Optional[np.ndarray] = None
        self.value_to_idx_: Dict[Any, int] = {}
        self.idx_to_value_: List[Any] = []
        self.q_: Optional[np.ndarray] = None

    def _build_vocab(self, records: Sequence[Sequence[Any]]):
        Vset = set()
        for row in records:
            if not row:
                Vset.add(None)
            else:
                Vset.update(row)
        self.idx_to_value_ = sorted(Vset, key=lambda x: (x is None, str(x)))
        self.value_to_idx_ = {v: i for i, v in enumerate(self.idx_to_value_)}

    def _ensure_vocab(self, records: Sequence[Sequence[Any]]):
        new_vals = []
        for row in records:
            row = row if row else [None]
            for v in row:
                if v not in self.value_to_idx_:
                    new_vals.append(v)
        if not new_vals:
            return
        start = len(self.idx_to_value_)
        for j, v in enumerate(new_vals, start=start):
            self.value_to_idx_[v] = j
            self.idx_to_value_.append(v)
        V_new = len(self.idx_to_value_)
        T = len(self.classes_)
        q_new = np.full((V_new, T), 1.0 / max(T, 1), dtype=np.float64)
        if self.q_ is not None:
            q_new[: self.q_.shape[0], :] = self.q_
        self.q_ = q_new

    def _menus_idx(self, records: Sequence[Sequence[Any]]) -> List[np.ndarray]:
        out = []
        for row in records:
            row = row if row else [None]
            out.append(np.array([self.value_to_idx_[v] for v in row], dtype=int))
        return out

    def fit(self, records: Sequence[Sequence[Any]], y: Sequence[Any]):
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        T = len(self.classes_)
        self.Py_ = np.bincount(y_idx, minlength=T).astype(np.float64)
        self._build_vocab(records)
        V = len(self.idx_to_value_)

        counts = np.zeros((V, T), dtype=np.float64)
        for i, row in enumerate(records):
            opts = row if row else [None]
            for v in opts:
                counts[self.value_to_idx_[v], y_idx[i]] += 1.0

        counts += float(self.smoothing)
        counts /= counts.sum(axis=1, keepdims=True)
        self.q_ = counts
        return self

    def _value_quality(self) -> np.ndarray:
        assert self.q_ is not None and self.Py_ is not None
        Pt = self.Py_ / self.Py_.sum()
        return (np.log(np.clip(self.q_, 1e-12, 1.0)) @ Pt)

    def _build_value_rows(self, records_idx: List[np.ndarray]) -> Dict[int, set]:
        value_rows: Dict[int, set] = defaultdict(set)
        for i, arr in enumerate(records_idx):
            for v in arr:
                value_rows[int(v)].add(i)
        return value_rows

    def _set_cover_catalog(
        self,
        value_rows: Dict[int, set],
        V: int,
        n_rows: int,
        K_target: int,
        s_val: np.ndarray,
    ) -> Tuple[List[int], int]:
        remaining = set(range(n_rows))
        S = []
        while remaining:
            best_v = max(
                range(V),
                key=lambda v: (len(value_rows.get(v, set()) & remaining), s_val[v]),
            )
            S.append(best_v)
            remaining -= value_rows.get(best_v, set())
        k_min = len(S)
        if K_target > k_min:
            extras = sorted(
                (v for v in range(V) if v not in S),
                key=lambda v: s_val[v],
                reverse=True,
            )[: K_target - k_min]
            S.extend(extras)
        S = sorted(S, key=lambda v: s_val[v], reverse=True)
        return S, k_min

    def _feasible_quotas(
        self,
        S: List[int],
        value_rows: Dict[int, set],
        piK: np.ndarray,
        n: int,
    ) -> np.ndarray:
        base_quota = _round_quota(piK, n)
        K = len(S)
        cap = np.array([len(value_rows.get(v, set())) for v in S], dtype=int)

        quota = np.minimum(base_quota, cap)
        deficit = int(n - quota.sum())
        if deficit > 0:
            for j in range(K):
                if deficit == 0:
                    break
                add = min(deficit, int(cap[j] - quota[j]))
                if add > 0:
                    quota[j] += add
                    deficit -= add

        if quota.sum() < n:
            need = int(n - quota.sum())
            for j in range(K):
                if need == 0:
                    break
                quota[j] += 1
                need -= 1
        return quota

    def _assign_with_quotas(
        self,
        records_idx: List[np.ndarray],
        S: List[int],
        quota: np.ndarray,
        s_val: np.ndarray,
    ) -> np.ndarray:
        n = len(records_idx)
        s_map = {v: s_val[v] for v in S}
        cap_left = quota.copy().astype(int)
        S_set = set(S)

        row_opts = []
        for arr in records_idx:
            opts = [v for v in arr if v in S_set]
            opts.sort(key=lambda v: s_map[v], reverse=True)
            row_opts.append(opts)

        order_rows = sorted(
            range(n),
            key=lambda i: (
                len(row_opts[i]),
                -sum(s_map[v] for v in row_opts[i]) if row_opts[i] else -1,
            ),
        )

        assign = -np.ones(n, dtype=int)
        for i in order_rows:
            for v in row_opts[i]:
                j = S.index(v)
                if cap_left[j] > 0:
                    assign[i] = v
                    cap_left[j] -= 1
                    break

        for i in range(n):
            if assign[i] >= 0:
                continue
            for v in row_opts[i]:
                assign[i] = v
                break
            if assign[i] < 0 and records_idx[i].size > 0:
                assign[i] = int(records_idx[i][0])

        return assign

    def predict(
        self,
        records: Sequence[Sequence[Any]],
        n_labels: Optional[int] = None,
        k_range: Tuple[int, int] = (2, 12),
    ) -> Dict[str, Any]:
        assert (
            self.q_ is not None and self.Py_ is not None and self.classes_ is not None
        ), "Llama fit() primero."
        self._ensure_vocab(records)
        records_idx = self._menus_idx(records)
        n = len(records_idx)
        V, T = self.q_.shape

        s_val = self._value_quality()
        value_rows = self._build_value_rows(records_idx)

        if n_labels is not None:
            K_candidates = [int(max(1, n_labels))]
        else:
            lo, hi = k_range
            lo = max(1, int(lo))
            hi = max(lo, int(hi))
            K_candidates = list(range(lo, min(hi, lo + 20) + 1))
            if int(math.sqrt(n) + 2) not in K_candidates:
                K_candidates.append(int(math.sqrt(n) + 2))

        best = None
        for K in K_candidates:
            S, k_min = self._set_cover_catalog(value_rows, V, n, K, s_val)
            if K < k_min:
                continue

            Py = self.Py_ / self.Py_.sum()
            piK = compress_distribution_to_K(Py, len(S))
            if piK.shape[0] != len(S):
                piK = np.resize(piK, len(S))
            quota = self._feasible_quotas(S, value_rows, piK, n)

            assign_idx = self._assign_with_quotas(records_idx, S, quota, s_val)

            s_pos = {v: j for j, v in enumerate(S)}
            hist = np.zeros(len(S), dtype=float)
            for v in assign_idx:
                if v in s_pos:
                    hist[s_pos[v]] += 1.0
            hist /= max(1, n)
            cv_diff = abs(_cv(hist) - _cv(Py))

            score = (-hist @ s_val[S]) + 0.05 * len(S) + 0.5 * cv_diff
            cand = (score, cv_diff, S, quota, assign_idx)
            if (best is None) or (cand[0] < best[0]):
                best = cand

        if best is None:
            raise RuntimeError("No feasible K found in the given range.")

        _, cv_diff, S, quota, assign_idx = best
        labels = [self.idx_to_value_[j] for j in assign_idx]

        Py = self.Py_ / self.Py_.sum()
        piK = compress_distribution_to_K(Py, len(S))
        if piK.shape[0] != len(S):
            piK = np.resize(piK, len(S))
        target_hist = piK.copy()
        s_pos = {v: j for j, v in enumerate(S)}
        actual_hist = np.zeros(len(S), dtype=float)
        for v in assign_idx:
            if v in s_pos:
                actual_hist[s_pos[v]] += 1.0
        actual_hist /= max(1, n)

        return {
            "labels": labels,
            "selected_values": [self.idx_to_value_[v] for v in S],
            "n_labels": len(S),
            "target_hist": target_hist,
            "actual_hist": actual_hist,
            "label_to_quota": {self.idx_to_value_[S[j]]: int(quota[j]) for j in range(len(S))},
            "cv_diff": float(cv_diff),
            "classes": self.classes_,
        }
