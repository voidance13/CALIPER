import unittest
import importlib.util
from pathlib import Path

class CompositionTests(unittest.TestCase):
    def test_branch_overhead_and_no_double_classifier(self):
        p=Path(__file__).with_name('compose_routed_table5_20260922.py')
        self.assertTrue(p.exists(), 'new composition implementation missing')
        spec=importlib.util.spec_from_file_location('compose',p)
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        fixed={'total_seconds':2.0}
        core={'total_seconds':3.0}
        clf={'total_seconds':3.2,'classifier_seconds':0.2,'classifier_branch':'local'}
        self.assertEqual(m.overhead('caliper_clf','local',fixed,core,clf),2.0)
        clf['classifier_branch']='collaborative'
        self.assertEqual(m.overhead('caliper_clf','collaborative',fixed,core,clf),3.0)
        self.assertEqual(m.overhead('caliper',None,fixed,core,clf),3.0)
        self.assertEqual(m.overhead('cloud_deep',None,fixed,core,clf),2.0)
        with self.assertRaises(ValueError): m.overhead('caliper_clf','local',fixed,core,clf)

if __name__=='__main__':unittest.main()
