from pathlib import Path

import click
import pandas as pd
import numpy as np

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--test-file', default='test.csv', help='Test CSV file')
def main(data_path, test_file):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    test_csv = data_directory.joinpath(test_file)
    gt_csv = data_directory.joinpath('ground_truth.csv')
    new_test_csv = data_directory.joinpath('new_test_data.csv')

    print('Reading files...')
    df_test = pd.read_csv(test_csv)
    mask_click_out = df_test["action_type"] == "clickout item"
    df_clicks = df_test[mask_click_out]

    mask_ground_truth = df_clicks["reference"].notnull()
    df_gt = df_clicks[mask_ground_truth]

    df_gt.to_csv(gt_csv, index=False)

    df_gt.loc[:, "reference"] = np.nan

    df_gt.to_csv(new_test_csv, index=False)

    print('finished')


if __name__ == '__main__':
    main()
