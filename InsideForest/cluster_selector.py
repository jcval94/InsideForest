from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple, Mapping
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import random
import math


def select_clusters(
    df_datos: pd.DataFrame,
    df_reglas: pd.DataFrame,
    keep_all_clusters: bool = True,
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

    reglas_info = []
    for _, row in df_reglas.iterrows():
        if row[('metrics', 'ponderador')] == 0:
            continue
        linf = row['linf'].dropna()
        lsup = row['lsup'].dropna()
        variables = linf.index.tolist()

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
                'linf': linf.to_dict(),
                'lsup': lsup.to_dict(),
                'ponderador': ponderador,
                'cluster': cluster_raw,
            }
        )

    for regla in reglas_info:
        variables = regla['variables']
        linf = regla['linf']
        lsup = regla['lsup']
        ponderador = regla['ponderador']
        cluster = regla['cluster']

        X_datos = df_datos[variables]
        condiciones = [
            (X_datos[var].to_numpy() >= linf[var]) & (X_datos[var].to_numpy() <= lsup[var])
            for var in variables
        ]
        if condiciones:
            cumple_regla = np.logical_and.reduce(condiciones)
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

    return clusters_datos, clusters_datos_all, ponderadores_datos_all


class MenuClusterSelector:
    """
    Selector de 'clústers' cuando X = records (solo menús por fila).
    Entrena con y para estimar q_v(y). En predicción elige un valor por fila
    maximizando un objetivo global: J = w_nmi * NMI + w_v * V_measure - λ * RegK.

    - fit(records_train, y): calcula q_v(y) (suavizado Laplace) y fija el vocabulario de valores.
    - predict(records, n_clusters=None): asigna un valor por fila SIN ver y, optimizando J
      por ascenso coordinado (greedy por mejoras de J). Si se da n_clusters=K, primero
      restringe a un catálogo S de tamaño K mediante cobertura+calidad y luego optimiza.
    """

    # =========================
    #   MÉTRICAS (explícitas)
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
        V-measure = armónica(homogeneidad, completitud) sobre C, Py, Pv.
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
        Regularizador global sobre el nº de valores usados:
          - Si target_K es None: Reg = lam * H(V)  (castiga alta entropía ⇒ menos valores efectivos).
          - Si target_K es int:  Reg = lam * (H(V) - log(target_K))^2  (empuja a ~K valores).
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
        # Añade valores no vistos en train con q_v uniforme (suavizado)
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
        Asigna 1 valor por fila maximizando J = w_nmi*NMI + w_v*V - lam_k*RegK,
        usando ascenso coordinado (greedy por mejoras).
        Si n_clusters=K, primero restringe a un catálogo S de tamaño K (cobertura+calidad).
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
    Asigna **un único valor por fila** optimizando dos objetivos con *peso idéntico*:

    • |distinct - n_clusters| →   acercarse al nº deseado de clusters
      (si `n_clusters` es `None`, se toma el mínimo posible de forma natural).

    • Desbalance absoluto    →   Σ |c_v – ideal| / n, donde `ideal = n / k`.
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
        """Mueve una fila a otra opción válida (aleatorio)."""
        i = rng.randrange(n)
        row = records[i]
        cur = assign[i]
        alt = [v for v in row if v != cur]
        if not alt:  # fila sin alternativas
            return assign
        new = assign[:]
        new[i] = rng.choice(alt)
        return new

    # Inicialización razonable ------------------------------------------
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
    Selecciona **un valor por fila** cumpliendo:
      • Si `n_clusters` es `None`  →  minimiza el nº de valores distintos.
      • Si `n_clusters` es un entero:
          – intenta devolver EXACTAMENTE ese nº de clusters, maximizando la suma de probabilidades.
          – si es imposible, usa el valor factible más próximo (`k_min` o `k_max`).
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

    # Paso 3: asignación greedy -----------------------------------------
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
