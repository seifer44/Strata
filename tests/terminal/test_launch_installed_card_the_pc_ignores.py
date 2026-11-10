"""The installed config names a card this PC's card list ignores: the choice cannot stay (fake rows, no hardware).

Measured on the Windows workstation: the app's install wrote "gpu": 0 into the coder config (`C:\\...\\Strata\\install\\strata-coder-iq1_m.json`,
backend hip, `gpus_asked` true, no `gpus_by`, no `hip_ordinal`), and a launch that named no card kept that 0 - the HIP ordinal of the
integrated Radeon - and the engine stopped on "GPU 0 (AMD Radeon(TM) Graphics, gfx1036) is not an architecture this Strata engine was
compiled for (gfx1200,gfx1201)". The card list (`strata_control/gpu_fit.usable_cards`) ignores that chip beside a card with memory of
its own, so the start now chooses again.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli.launch import catalog as C, runner as R, settings as S      # noqa: E402


class CardChoice(unittest.TestCase):
    def computer(self, cards, backend="amd"):
        hw = {"ram_gb": 96.0, "os": "Windows", "cpu": {"avx2": True}, "gpus": cards}
        cat = C.Catalog(root=Path(tempfile.mkdtemp()), hardware=hw)
        cat.engine_backend = lambda: backend
        cat.engine_vendor = lambda hw=None: backend
        return cat

    def workstation(self):
        # The mixed PC: NVIDIA in the adapter list, the AMD make numbered from 0 (setup.py amd_gpus_windows skips NVIDIA)
        return [
            {"index": 0, "name": "NVIDIA GeForce GTX 1660 Ti", "vendor": "nvidia", "vram_gb": 5.7},
            {"index": 0, "name": "AMD Radeon(TM) Graphics", "vendor": "amd", "vram_gb": 12.2,
             "integrated": True, "arch": "gfx1036"},
            {"index": 1, "name": "AMD Radeon RX 9070 XT", "vendor": "amd", "vram_gb": 15.9, "arch": "gfx1201"},
        ]

    def test_the_integrated_ordinal_the_app_recorded_is_not_a_card_that_counts(self):
        cat = self.computer(self.workstation())
        keep, ignored = cat.computer().usable_gpus, [c for c in cat.computer().gpus]
        self.assertEqual([int(c["index"]) for c in keep], [1])           # the discrete card alone counts
        self.assertNotIn(0, {int(c["index"]) for c in keep})             # the integrated chip is left out

    def test_a_stale_integrated_choice_moves_to_the_card_that_counts(self):
        cat = self.computer(self.workstation())
        base = {"gpu": 0, "gpus_asked": True}                            # what the app's install wrote
        got = R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat, base=base)
        self.assertEqual(got, [1])                                      # chosen again, not kept
        cfg = {}
        R.apply_cards(cfg, got)
        self.assertEqual(cfg["gpu"], 1)
        self.assertTrue(cfg["gpus_asked"])
        self.assertNotIn("hip_ordinal", cfg)                            # apply_cards takes it away, as before

    def test_a_number_that_names_no_card_at_all_moves_too(self):
        cat = self.computer(self.workstation())
        got = R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                 base={"gpu": 2, "gpus_asked": True})
        self.assertEqual(got, [1])

    def test_a_choice_that_names_a_card_that_counts_stays(self):
        cat = self.computer(self.workstation())
        self.assertIsNone(R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                             base={"gpu": 1, "gpus_asked": True}))

    def test_nothing_recorded_still_names_nothing(self):
        cat = self.computer(self.workstation())
        self.assertIsNone(R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                             base={"gpu": 1, "gpus_asked": False}))

    def test_roles_a_person_set_are_never_moved_over_a_stale_choice(self):
        cat = self.computer(self.workstation())
        self.assertIsNone(R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                             base={"gpu": 0, "gpus_asked": True, "gpu_roles": {0: "main"}}))

    def test_two_cards_that_count_and_a_stale_choice_use_the_role_rule(self):
        cards = self.workstation() + [{"index": 2, "name": "AMD Radeon RX 7900 XT", "vendor": "amd",
                                       "vram_gb": 19.6, "arch": "gfx1100"}]
        cat = self.computer(cards)
        self.assertEqual(sorted(int(c["index"]) for c in cat.computer().usable_gpus), [1, 2])
        got = R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                 base={"gpu": 0, "gpus_asked": True})
        self.assertTrue(got)
        self.assertEqual(sorted(got), [1, 2])
        self.assertNotIn(0, got)

    def test_a_pc_whose_only_device_is_the_chip_keeps_its_card(self):
        cards = [{"index": 0, "name": "AMD Radeon (Strix Halo)", "vendor": "amd", "vram_gb": 0.0,
                  "unified_gb": 96.0, "uma": True}]
        cat = self.computer(cards)
        self.assertEqual([int(c["index"]) for c in cat.computer().usable_gpus], [0])
        self.assertIsNone(R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                             base={"gpu": 0, "gpus_asked": True}))


    def notices(self, cards, base, chosen, backend="amd"):
        cat = self.computer(cards, backend)
        run = R.Runner(S.LaunchSettings(family="coder", model="IQ1_M"), catalog=cat)
        run.say_remembered_cards(base, chosen)
        return run.notices

    def test_the_note_says_the_card_that_moved_and_not_the_one_left_behind(self):
        got = self.notices(self.workstation(), {"gpu": 0, "gpus_asked": True}, [1])
        self.assertEqual(len(got), 1)
        self.assertIn("installed config names card 0", got[0])
        self.assertIn("this start uses card 1 instead", got[0])
        self.assertIn("--gpu 1", got[0])
        self.assertNotIn("stays on", got[0])          # the old line described a run that does not happen

    def test_a_stale_choice_with_two_cards_that_counts_names_both(self):
        cards = self.workstation() + [{"index": 2, "name": "AMD Radeon RX 7900 XT", "vendor": "amd",
                                       "vram_gb": 19.6, "arch": "gfx1100"}]
        cat = self.computer(cards)
        chosen = R.cards_for_launch(S.LaunchSettings(family="coder", model="IQ1_M"), cat,
                                    base={"gpu": 0, "gpus_asked": True})
        got = self.notices(cards, {"gpu": 0, "gpus_asked": True}, chosen)
        self.assertEqual(len(got), 1)
        self.assertIn("installed config names card 0", got[0])
        self.assertIn("--gpu ", got[0])
        self.assertNotIn("stays on", got[0])

    def test_a_choice_that_counts_still_says_nothing_on_a_one_card_pc(self):
        self.assertEqual(self.notices(self.workstation(), {"gpu": 1, "gpus_asked": True}, None), [])

    def test_an_unrecorded_config_says_nothing(self):
        self.assertEqual(self.notices(self.workstation(), {"gpu": 1, "gpus_asked": False}, None), [])

if __name__ == "__main__":
    unittest.main()
