import importlib.util
from pathlib import Path
import unittest


class LiveTimingTests(unittest.TestCase):
    def test_reranker_binding_rejects_implicit_multi_gpu(self):
        spec = importlib.util.spec_from_file_location('live', Path(__file__).with_name('live_table5_20260922.py'))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertTrue(hasattr(m, 'single_gpu_reranker'), 'device binding guard missing')
        # Mirrors installed FlagEmbedding: singular device is swallowed by kwargs.
        class RerankerBoundary:
            def __init__(self, path, use_fp16=False, devices=None, **kwargs):
                self.target_devices = devices or ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        reranker = m.single_gpu_reranker(RerankerBoundary, 'model', 'cuda:2')
        self.assertEqual(reranker.target_devices, ['cuda:2'])
        class IgnoredBinding(RerankerBoundary):
            def __init__(self, *args, **kwargs):
                self.target_devices = ['cuda:0', 'cuda:1']
        with self.assertRaises(RuntimeError):
            m.single_gpu_reranker(IgnoredBinding, 'model', 'cuda:2')

    def test_complete_chain_and_classifier_branch(self):
        path = Path(__file__).with_name('live_table5_20260922.py')
        self.assertTrue(path.exists(), 'integrated timing runner missing')
        spec = importlib.util.spec_from_file_location('live', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # External GPU/API boundaries replaced; clock advances only inside stages.
        events, now = [], [0.0]
        def retrieval(q, method):
            events.append(('retrieval', method)); now[0] += 2
            return {'context': ['new evidence'], 'branch': 'local', 'source_route': 'ms'}
        def generation(q, context, method, branch):
            self.assertEqual(context, ['new evidence'])
            events.append(('generation', method, branch)); now[0] += 3
            return {'prediction': 'answer'}
        result = m.measure_one('q', 'caliper_clf', retrieval, generation, lambda: now[0])
        self.assertEqual(result['seconds'], 5)
        self.assertEqual(result['prediction'], 'answer')
        self.assertEqual(events, [('retrieval', 'caliper_clf'), ('generation', 'caliper_clf', 'local')])


if __name__ == '__main__':
    unittest.main()
