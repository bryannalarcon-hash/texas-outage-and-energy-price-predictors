import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
from forecast_dispatch import solve, reserve_curve, ETA

class DispatchTests(unittest.TestCase):
    def test_realized_cycle_and_terminal_inventory(self):
        c,d,s,f=solve([10.,100.],np.full(3,5.))
        self.assertGreater(c[0],1.24)
        self.assertAlmostEqual(d[1],.9*c[0],places=7)
        self.assertAlmostEqual(s[-1],15.)
        self.assertGreater(np.dot([10.,100.],d-c)/1000, .09)
        self.assertLess(float(np.max(c*d)),1e-8)
    def test_flat_positive_prices_do_not_pay_for_losses(self):
        c,d,s,f=solve([30.]*8,np.full(9,5.))
        self.assertLess(float(c.sum()+d.sum()),1e-7)
    def test_reserve_is_never_invented_beyond_complete_window(self):
        origin=datetime(2025,7,1,12,tzinfo=timezone.utc)
        target,covered=reserve_curve(np.full(24,.01),origin,[origin,origin+timedelta(hours=19)])
        self.assertAlmostEqual(target[0],7.4)
        self.assertEqual(target[1],5.)
        self.assertEqual(list(covered),[True,False])
    def test_negative_price_schedule_obeys_physics(self):
        c,d,s,f=solve([-100.,-10.,200.,10.],np.full(5,5.))
        np.testing.assert_allclose(s[1:],s[:-1]+ETA*c-d/ETA,atol=1e-7)
        self.assertTrue(np.all(s>=f-1e-7))

if __name__=='__main__': unittest.main()
