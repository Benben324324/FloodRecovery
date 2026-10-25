"""Checks for safeguards that affect the scientific interpretation."""
import unittest
import numpy as np
import pandas as pd
from src.lights import read_pair, return_date, trailing_median, load_inputs, analyze
from src.radar import metrics, load_split, SIZES


class AnalysisChecks(unittest.TestCase):
    def test_flag_alignment_and_categories(self):
        p = read_pair('Zhengzhou', 2021)
        self.assertEqual(p['light'].shape, p['flags'].shape)
        self.assertTrue(p['flags_reprojected'])
        self.assertTrue(np.isin(p['flags'], [0, 10, 11, 20, 21, 30, 31]).all())
        self.assertFalse(np.any(p['observed'] & ~p['valid']))

    def test_sparse_dates_do_not_establish_return(self):
        s = pd.Series([.7, .75, .72, 1.1, 1.1, 1.1],
                      index=pd.to_datetime(['2021-07-17','2021-07-18','2021-07-19',
                                            '2021-08-01','2021-08-09','2021-08-17']))
        date, status = return_date(s)
        self.assertIsNone(date)

    def test_three_close_supported_dates_establish_return(self):
        dates = pd.date_range('2021-07-17', '2021-08-04')
        s = pd.Series(np.r_[np.full(8, .7), np.full(len(dates)-8, 1.05)], index=dates)
        date, _ = return_date(s)
        self.assertIsNotNone(date)
        self.assertGreater(date, pd.Timestamp('2021-07-24'))

    def test_no_decline_has_no_recovery_date(self):
        s = pd.Series(1.1, index=pd.date_range('2021-07-17', '2021-08-10'))
        date, status = return_date(s)
        self.assertIsNone(date)
        self.assertEqual(status, 'No supported threshold decline')

    def test_window_support_uses_calendar_days(self):
        s = pd.Series([1, 1, 1], index=pd.to_datetime(['2021-06-01','2021-06-11','2021-06-21']))
        self.assertTrue(trailing_median(s).isna().all())

    def test_radar_metric_reference_case(self):
        m = metrics([8, 2, 4, 6])
        self.assertAlmostEqual(m['iou'], 8/14)
        self.assertAlmostEqual(m['f1'], 16/22)

    def test_radar_split_pairs(self):
        for split, n in SIZES.items():
            self.assertEqual(len(load_split(split)), n)


if __name__ == '__main__':
    unittest.main()
