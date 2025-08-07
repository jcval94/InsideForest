import numpy as np
import logging

logger = logging.getLogger(__name__)


class Labels:
    def round_values(self, values):
        variance = np.var(values)
        if variance >= 0.01 and len(values) > 1:
            return [round(val, 2) for val in values]
        else:
            return ['{:.2e}'.format(val) for val in values]

    def custom_round(self, number):
        if abs(number) > 100 or abs(number - int(number)) < 1e-10:
            return int(number)
        elif abs(number) < 0.01:
            return "{:.2e}".format(number)
        else:
            return round(number, 3)

    def get_intervals(self, interval_df):
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
        height_columns = [col for col in df.columns if 'altura' in col[1]]
        return df.drop(height_columns, axis=1)

    def get_branch(self, df, sub_df, row_index):
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
        labels_list = []
        for branch_index in range(num_branches - 1):
            if branch_index >= len(range_dataframes):
                continue
            current_range_df = range_dataframes[branch_index].copy()
            current_range_df = current_range_df[
                [(a, b) for a, b in current_range_df.columns if 'altura' != b]
            ]
            interval_descriptions = self.get_intervals(
                current_range_df.head(max_labels)
            )
            try:
                branch_dfs = [
                    self.get_branch(df, current_range_df, i)
                    for i in range(0, max_labels + 1)
                ]
                score_population = [
                    (x[target_var].mean(), x[target_var].count())
                    for x in branch_dfs
                    if x is not None
                ]
                target_population = [
                    x[x[target_var] == 0]
                    for x in branch_dfs
                    if x is not None
                ]
            except KeyError as exc:
                logger.exception(
                    "Missing columns when obtaining labels: %s",
                    exc,
                )
                continue
            if len(target_population) == 0:
                continue
            labels_dict = {
                description: [score, population]
                for population, score, description in zip(
                    target_population,
                    score_population,
                    interval_descriptions,
                )
                if population.shape[0] > 0
            }
            labels_list.append(labels_dict)

        return labels_list

