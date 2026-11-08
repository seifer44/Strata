"""A make named for a start reaches setup.py's own --backend in the install job (#4075: an AMD card
beside an NVIDIA card on a Windows PC). Without it the Terminal numbers the AMD card and the install
job validates that number against its own card list, whose pick is the NVIDIA card first when both
makes are in the PC (setup.py:10261/10262), so every card the Terminal named is refused as "not found
on this PC". Only a make the person named (STRATA_ENGINE_BACKEND, set by `strata launch --backend` at
launch/cmd.py:863) is passed: the release's default argv is unchanged (checked against a plain
v0.2.12 tree, run for run), and a saved make choice keeps setup.py's own pick."""

import os
import unittest
from unittest import mock

from strata_cli.launch import runner as R
from strata_cli.launch.runner import Runner
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_control import engine_component as ec
from tests.terminal.launch_support import HW_12, Home, catalog

#: the reporter's PC (2026-10-28 / 2026-11-08): a GTX 1660 Ti, an integrated Radeon and an RX 9070 XT,
#: in display-adapter order, the numbering the Terminal's card screen shows.
HW_MIXED = {**HW_12, "gpus": [
    {"index": 0, "name": "NVIDIA GeForce GTX 1660 Ti", "vram_gb": 6.0, "usable": True, "vendor": "nvidia", "arch": 75},
    {"index": 1, "name": "AMD Radeon(TM) Graphics", "vram_gb": 12.2, "usable": True, "vendor": "amd", "arch": "gfx1036", "uma": True},
    {"index": 2, "name": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, "usable": True, "vendor": "amd", "arch": "gfx1201"}]}


class AStartOnThatPc(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def argv(self, make=None, hw=HW_MIXED):
        """The setup.py argv the install job builds, with the make in or out of the environment."""
        saved = os.environ.pop(ec.ENV_BACKEND, None)
        if make:
            os.environ[ec.ENV_BACKEND] = make
        try:
            s = S.LaunchSettings(family="coder", model="IQ1_M", context=131072, vision="off")
            with mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(hw, disk_free_gb=500.0)):
                return Runner(s, catalog=catalog(hw)).setup_args()
        finally:
            if saved is not None:
                os.environ[ec.ENV_BACKEND] = saved

    def test_no_make_keeps_setup_s_own_pick(self):
        self.assertNotIn("--backend", self.argv(), "the release's argv: setup.py picks the make itself")

    def test_the_amd_ask_reaches_the_amd_branch(self):
        args = self.argv("hip")
        self.assertIn("--backend", args, "the make the Terminal used must reach the job that installs the model")
        self.assertEqual(args[args.index("--backend") + 1], "hip", "setup.py's own name for the AMD make")

    def test_a_build_name_and_a_make_word_are_the_same_branch(self):
        for ask in ("hip", "hip-rx9070", "amd"):
            args = self.argv(ask)
            self.assertEqual(args[args.index("--backend") + 1], "hip",
                             f"{ask} names the hip family; which build the card needs stays setup's")

    def test_the_other_makes_use_their_own_names(self):
        for ask, want in (("nvidia", "cuda"), ("cuda", "cuda"), ("intel", "sycl"), ("sycl", "sycl")):
            args = self.argv(ask)
            self.assertEqual(args[args.index("--backend") + 1], want)

    def test_a_name_that_is_not_a_make_is_not_passed_on(self):
        self.assertNotIn("--backend", self.argv("swift"),
                         "setup.py takes cuda, hip or sycl only (setup.py:10093): an unknown word is its "
                         "own to report, not an argument it will refuse")

    def test_the_make_is_asked_on_both_cards_lists(self):
        """An NVIDIA-only PC asked for its own make also reaches setup.py, and keeps its card numbers."""
        args = self.argv("nvidia", HW_12)
        self.assertEqual(args[args.index("--backend") + 1], "cuda")

    def test_a_saved_choice_alone_changes_nothing(self):
        """A PC that saved a make choice keeps setup.py's own pick: that path is #4075's scope question,
        not something to change here."""
        with mock.patch.object(ec, "saved_backend", lambda environ=None: "hip"):
            self.assertNotIn("--backend", self.argv())

    def test_the_mac_setup_takes_no_backend(self):
        with mock.patch.object(R.sys, "platform", "darwin"):
            args = self.argv("metal")
        self.assertNotIn("--backend", args, "the Mac's path keeps its own setup (setup.py:10093's choices)")
