import numpy as np
import logging

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
        if abs(number) > 100 or abs(number - int(number)) < 1e-10:
            return int(number)
        elif abs(number) < 0.01:
            return "{:.2e}".format(number)
        else:
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
        interval_df = interval_df.applymap(self.custom_round)
        interval_descriptions = []
        for row_index in range(len(interval_df)):
            row_result = []
            for col in interval_df[['linf']].columns:
                lower_value = interval_df.iloc[row_index][('linf', col[-1])]
                upper_value = interval_df.iloc[row_index][('lsup', col[-1])]
                if lower_value == upper_value:
                    continue
                row_result.append(
                    f"{col[-1]} between {lower_value} and {upper_value}"
                )
            joined_descriptions = " | ".join(row_result)
            interval_descriptions.append(joined_descriptions)
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
        labels_list = []
        for branch_index in range(num_branches - 1):
            if branch_index >= len(range_dataframes):
                continue
            current_range_df = range_dataframes[branch_index]
            current_range_df = current_range_df[
                [(a, b) for a, b in current_range_df.columns if "altura" != b]
            ]
            interval_descriptions = self.get_intervals(
                current_range_df.head(max_labels)
            )
            num_rows = len(interval_descriptions)
            if num_rows == 0:
                continue
            try:
                variables = current_range_df["linf"].columns
                data_matrix = df[variables].to_numpy(copy=False)
                lower_bounds = current_range_df["linf"].to_numpy(copy=False)[
                    :num_rows
                ]
                upper_bounds = current_range_df["lsup"].to_numpy(copy=False)[
                    :num_rows
                ]
                masks = np.all(
                    (data_matrix[None, :, :] <= upper_bounds[:, None, :])
                    & (data_matrix[None, :, :] > lower_bounds[:, None, :]),
                    axis=2,
                )
                target_array = df[target_var].to_numpy(copy=False)
            except KeyError as exc:
                logger.exception(
                    "Missing columns when obtaining labels: %s",
                    exc,
                )
                continue
            labels_dict = {}
            for mask, description in zip(masks, interval_descriptions):
                if not mask.any():
                    continue
                population_mask = mask & (target_array == 0)
                if not population_mask.any():
                    continue
                score = (target_array[mask].mean(), int(mask.sum()))
                population = df.loc[population_mask]
                labels_dict[description] = [score, population]
            if labels_dict:
                labels_list.append(labels_dict)

        return labels_list

