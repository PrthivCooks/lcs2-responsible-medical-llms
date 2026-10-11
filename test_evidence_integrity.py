"""Offline integrity checks for the committed LCS2 evidence. No GPU or credentials."""
import collections
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'llama32': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen25': 'Qwen/Qwen2.5-3B-Instruct',
}
CONDITIONS = (
    'core_clean', 'misleading', 'race_black', 'race_white',
    'insurance_uninsured', 'insurance_private', 'residence_rural',
    'residence_urban', 'target_injection', 'canary_attack',
    'mitigation_target', 'mitigation_clean', 'mitigation_misleading',
)


def read_jsonl(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


class EvidenceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads((ROOT / 'metadata' / 'run_config.json').read_text())
        cls.config = cls.meta['config']

    def test_frozen_protocol(self):
        cfg = self.config
        self.assertEqual(cfg['dataset_repo'], 'Johndfm/MedMisBench-MentalHealth')
        self.assertEqual(cfg['dataset_revision'], '9757095')
        self.assertEqual(cfg['seed'], 42)
        self.assertEqual(cfg['quantization'], 'float16')
        self.assertEqual(cfg['decoding'], {'do_sample': False, 'num_beams': 1})
        self.assertEqual(cfg['core'], list(range(200)))
        self.assertEqual(cfg['stress'], list(range(60)))
        self.assertEqual(cfg['interpret'], list(range(40)))
        self.assertEqual(cfg['condition_list'], list(CONDITIONS))
        self.assertEqual(set(cfg['model_ids']), set(MODELS.values()))

    def test_generation_and_interpretability_matrices(self):
        for slug, model in MODELS.items():
            with self.subTest(model=slug):
                records = read_jsonl(ROOT / 'records' / f'{slug}.jsonl')
                interpretation = read_jsonl(ROOT / 'interpretability' / f'{slug}.jsonl')
                self.assertEqual(len(records), 920)
                self.assertEqual(len(interpretation), 40)
                self.assertEqual(collections.Counter(r['condition'] for r in records),
                                 {'core_clean': 200, **{c: 60 for c in CONDITIONS if c != 'core_clean'}})
                self.assertEqual(len({(r['condition'], r['row_index']) for r in records}), 920)
                self.assertEqual({r['row_index'] for r in interpretation}, set(range(40)))
                sha = self.config['model_shas'][model]
                index = {(r['condition'], r['row_index']): r for r in records}
                for r in records:
                    self.assertEqual(r['model'], model)
                    self.assertEqual(r['model_revision'], sha)
                    self.assertEqual(r['run_fingerprint'], self.meta['fingerprint'])
                    self.assertIsNone(r['error'])
                    self.assertTrue(0 <= r['row_index'] < (200 if r['condition'] == 'core_clean' else 60))
                    self.assertEqual(r['correct'], r['parsed_answer'] == r['gold_answer'])
                    self.assertIn('raw_model_output', r)
                for cond in CONDITIONS:
                    for i in range(60):
                        self.assertEqual(index[cond, i]['dataset_id'], index['core_clean', i]['dataset_id'])
                        self.assertEqual(index[cond, i]['gold_answer'], index['core_clean', i]['gold_answer'])
                for r in interpretation:
                    self.assertEqual(r['run_fingerprint'], self.meta['fingerprint'])
                    self.assertEqual(r['model_revision'], sha)
                    self.assertEqual(r['dataset_id'], index['core_clean', r['row_index']]['dataset_id'])
                    self.assertEqual(r['gold_answer'], index['core_clean', r['row_index']]['gold_answer'])
                    self.assertIn('faithfulness_advantage', r)

    def test_cross_model_question_alignment(self):
        maps = {}
        for slug in MODELS:
            rows = read_jsonl(ROOT / 'records' / f'{slug}.jsonl')
            maps[slug] = {r['row_index']: (r['dataset_id'], r['gold_answer'])
                          for r in rows if r['condition'] == 'core_clean'}
        self.assertEqual(maps['llama32'], maps['qwen25'])

    def test_snapshot_checksum_is_recorded(self):
        expected = self.config['dataset_sha256']
        self.assertEqual(len(expected), 64)
        self.assertTrue(all(x in '0123456789abcdef' for x in expected))
        # The dataset itself is downloaded in the GPU notebook and not redistributed here.


if __name__ == '__main__':
    unittest.main()
