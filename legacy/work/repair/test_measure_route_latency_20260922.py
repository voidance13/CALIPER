import unittest
import measure_route_latency_20260922 as m


class ProtocolTests(unittest.TestCase):
    def test_only_authorized_route_mismatches_may_replay(self):
        self.assertEqual(m.resolve_route('musique', 72, 'hp', 'ms'), ('ms', True))
        self.assertEqual(m.resolve_route('musique', 86, 'hp', 'ms'), ('ms', True))
        self.assertEqual(m.resolve_route('musique', 1, 'hp', 'hp'), ('hp', False))
        for args in [('musique', 73, 'hp', 'ms'), ('hotpotqa', 72, 'hp', 'ms'), ('musique', 72, 'tw', 'ms')]:
            with self.assertRaises(ValueError):
                m.resolve_route(*args)

    def test_frozen_row_id_not_first_duplicate_supplies_candidates(self):
        rows = [{'question': 'q', 'ctxs': [{'sentences': ['wrong']} ]},
                {'question': 'q', 'ctxs': [{'sentences': ['right']} ]}]
        self.assertEqual(m.frozen_texts('q', 1, rows, False), ['right'])
        dev = [{'question': 'q', 'paragraphs': [{'paragraph_text': 'a. b'}]}]
        self.assertEqual(m.frozen_texts('q', 0, dev, True), ['a', 'b'])
        with self.assertRaises(ValueError):
            m.frozen_texts('missing', 0, rows, False)

    def test_same_route_preserves_per_question_checkpoint(self):
        rows = {'a': {'predicted_route': 'ms', 'model': {'embedder': 'old', 'reranker': 'r1'}},
                'b': {'predicted_route': 'ms', 'model': {'embedder': 'new', 'reranker': 'r2'}}}
        self.assertEqual(m.model_for(rows, 'a', 'ms'), ('old', 'r1'))
        self.assertEqual(m.model_for(rows, 'b', 'ms'), ('new', 'r2'))
        with self.assertRaises(ValueError):
            m.model_for(rows, 'a', 'hp')

    def test_pair_duplicate_questions_by_occurrence_not_last_row(self):
        rows = [{'question': 'q', 'id': 1}, {'question': 'q', 'id': 2}]
        self.assertEqual([r['id'] for r in m.align_rows(['q', 'q'], rows)], [1, 2])
        with self.assertRaises(ValueError):
            m.align_rows(['missing'], rows)

    def test_dual_context_caps_and_deduplicates(self):
        self.assertEqual(m.contexts('musique', ['a','b','c','d','e','f'], [' A ', 'x','y','z']),
                         ['a','b','c','d','e','x','y'])
        self.assertEqual(m.contexts('hotpotqa', ['a'], ['x']), ['a'])

    def test_classifier_is_not_added_twice_to_generation(self):
        self.assertEqual(m.composable_overhead({'total_seconds': 3, 'classifier_seconds': .5}, True), 2.5)
        self.assertEqual(m.composable_overhead({'total_seconds': 3, 'classifier_seconds': .5}, False), 3)


if __name__ == '__main__':
    unittest.main()
