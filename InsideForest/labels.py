import logging
import math
from numbers import Real

import numpy as np

logger = logging.getLogger(__name__)


class Labels:
    """Helper methods for converting tree ranges into readable labels."""

    def round_values(self, values):
        """Round numeric values using variance-aware precision.

        Parameters
        ----------
        values : Sequence[float]
            Numeric values to format.

        Returns
        -------
        list of float or str
            Values rounded to two decimals when variance is high; otherwise
            formatted using scientific notation.
        """
        variance = np.var(values)
        if variance >= 0.01 and len(values) > 1:
            return [round(val, 2) for val in values]
        else:
            return ['{:.2e}'.format(val) for val in values]

    @staticmethod
    def _values_equal(value_a, value_b):
        """Return ``True`` when two rounded values should be treated as equal."""

        if isinstance(value_a, Real) and isinstance(value_b, Real):
            return math.isclose(float(value_a), float(value_b), rel_tol=0, abs_tol=1e-9)
        return value_a == value_b

    def custom_round(self, number):
        """Round a number according to its magnitude.

        Parameters
        ----------
        number : float
            Value to round.

        Returns
        -------
        int | float | str
            Integers are returned for very large or nearly integral values,
            scientific notation for very small magnitudes, and otherwise the
            number rounded to three decimals.
        """
        if number is None:
            return None

        if isinstance(number, bool):
            return int(number)

        if not isinstance(number, Real):
            return number

        if isinstance(number, float) and math.isnan(number):
            return number

        if isinstance(number, float) and math.isinf(number):
            return number

        magnitude = abs(number)
        if magnitude >= 100 or math.isclose(number, round(number), rel_tol=0, abs_tol=1e-10):
            return int(round(number))

        if 0 < magnitude < 0.01:
            return "{:.2e}".format(number)

        return round(number, 3)

    def get_intervals(self, interval_df):
        """Generate textual descriptions from interval bounds.

        Parameters
        ----------
        interval_df : pd.DataFrame
            DataFrame containing ``linf`` and ``lsup`` columns for each
            variable.

        Returns
        -------
        list of str
            One description per row combining variable names and their
            respective lower and upper limits.
        """
        interval_df = self.drop_height_columns(interval_df)
        if getattr(interval_df.columns, "nlevels", 1) < 2:
            raise ValueError("interval_df must have a two-level column MultiIndex")

        rounded_df = interval_df.map(self.custom_round)
        interval_descriptions = []
        linf_columns = [col for col in rounded_df.columns if col[0] == "linf"]

        for row_index in range(len(rounded_df)):
            row_result = []
            for _, variable in linf_columns:
                try:
                    lower_value = rounded_df.iloc[row_index][("linf", variable)]
                    upper_value = rounded_df.iloc[row_index][("lsup", variable)]
                except KeyError:
                    continue
                if self._values_equal(lower_value, upper_value):
                    continue
                row_result.append(
                    f"{variable} between {lower_value} and {upper_value}"
                )
            interval_descriptions.append(" | ".join(row_result))

        return interval_descriptions

    def drop_height_columns(self, df):
        """Remove columns whose second-level name contains ``'altura'``.

        Parameters
        ----------
        df : pd.DataFrame
            Multi-indexed DataFrame from which to drop height-related columns.

        Returns
        -------
        pd.DataFrame
            DataFrame without the height columns.
        """
        height_columns = [col for col in df.columns if 'altura' in col[1]]
        return df.drop(height_columns, axis=1)

    def get_branch(self, df, sub_df, row_index):
        """Return the subset of ``df`` satisfying bounds at ``row_index``.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataset to filter.
        sub_df : pd.DataFrame
            DataFrame with ``linf`` and ``lsup`` bounds for each variable.
        row_index : int
            Row in ``sub_df`` specifying the interval to apply.

        Returns
        -------
        pd.DataFrame or None
            Filtered DataFrame matching the bounds or ``None`` if
            ``row_index`` exceeds the number of rows in ``sub_df``.
        """
        sub_df.reset_index(inplace=True, drop=True)

        if not set(sub_df.columns.get_level_values(1)).issubset(df.columns):
            missing = set(sub_df.columns.get_level_values(1)) - set(df.columns)
            raise KeyError(
                f"Columns {missing} do not exist in the main DataFrame"
            )

        if row_index >= len(sub_df):
            return None
        lower_bounds = sub_df.loc[row_index, 'linf'].copy()
        upper_bounds = sub_df.loc[row_index, 'lsup'].copy()
        variables = list(upper_bounds.index)

        # Early exit when there are no variables to filter on.
        # Returning an empty DataFrame prevents index errors when constructing
        # boolean conditions on an empty list of variables.
        if len(variables) == 0:
            return df.iloc[0:0]

        conditions = [
            (df[var] <= upper_bounds[var]) & (df[var] > lower_bounds[var])
            for var in variables
        ]

        combined_condition = conditions[0]
        if len(conditions) > 1:
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition

        return df[combined_condition]

    def get_labels(
        self,
        range_dataframes,
        df,
        target_var,
        max_labels=9,
        num_branches=10,
    ):
        """Generate label descriptions and statistics for tree branches.

        Parameters
        ----------
        range_dataframes : Sequence[pd.DataFrame]
            List of DataFrames describing variable intervals for each branch.
        df : pd.DataFrame
            Original dataset.
        target_var : str
            Target column used to compute scores and populations.
        max_labels : int, default 9
            Maximum number of interval descriptions per branch.
        num_branches : int, default 10
            Number of branches from ``range_dataframes`` to evaluate.

        Returns
        -------
        list of dict
            A list where each element is a dictionary mapping an interval
            description to ``[score, population]``. ``score`` contains the mean
            target value and count of observations, while ``population`` is the
            subset of ``df`` satisfying the interval.
        """
        if target_var not in df.columns:
            raise KeyError(f"Target column '{target_var}' not present in DataFrame")

        labels_list = []
        target_array = df[target_var].to_numpy(copy=False)

        branch_limit = min(max(num_branches - 1, 0), len(range_dataframes))
        for branch_index in range(branch_limit):
            current_range_df = range_dataframes[branch_index]
            if current_range_df is None or getattr(current_range_df, "empty", False):
                continue

            cleaned_range_df = self.drop_height_columns(current_range_df)
            if getattr(cleaned_range_df.columns, "nlevels", 1) < 2:
                continue

            limited_df = cleaned_range_df.head(max_labels)
            interval_descriptions = self.get_intervals(limited_df)
            valid_rows = [idx for idx, desc in enumerate(interval_descriptions) if desc]
            if not valid_rows:
                continue

            try:
                variables = list(limited_df["linf"].columns)
            except KeyError:
                logger.exception("Missing 'linf' columns when obtaining labels")
                continue

            if not variables:
                continue

            try:
                data_matrix = df[variables].to_numpy(copy=False)
            except KeyError as exc:
                logger.exception(
                    "Missing predictor columns when obtaining labels: %s",
                    exc,
                )
                continue

            try:
                lower_bounds = limited_df["linf"].iloc[valid_rows].to_numpy(copy=False)
                upper_bounds = limited_df["lsup"].iloc[valid_rows].to_numpy(copy=False)
            except KeyError as exc:
                logger.exception(
                    "Missing bound columns when obtaining labels: %s",
                    exc,
                )
                continue

            masks = np.all(
                (data_matrix[None, :, :] <= upper_bounds[:, None, :])
                & (data_matrix[None, :, :] > lower_bounds[:, None, :]),
                axis=2,
            )

            labels_dict = {}
            for mask, row_index in zip(masks, valid_rows):
                if not mask.any():
                    continue

                description = interval_descriptions[row_index]
                target_slice = target_array[mask]
                score = (float(np.nanmean(target_slice)), int(mask.sum()))
                population = df.loc[mask]
                labels_dict[description] = [score, population]

            if labels_dict:
                labels_list.append(labels_dict)

        return labels_list

