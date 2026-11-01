"""#4013 on a mixed-make PC: `strata update --backend hip` on a PC that also has an NVIDIA adapter.
The probe runs with the auto-detected backend (CUDA), so the AMD card is never probed and the HIP
builds are refused with "the card was not recognised". Asking for a make must probe that make.

The PC is the reporter's own output (2026-10-28): a GTX 1660 Ti (compute capability 7.5, 6 GB), an
RX 9070 XT (gfx1201, 16 GB) and an integrated Radeon. The registry lists all three, and the code maps
the discrete card by its PCI id (DEV_7550 = 0x7550, in WIN_AMD_DEVICE_ARCH)."""
from __future__ import annotations

import unittest
from unittest import mock
from contextlib import ExitStack

from strata_control import engine_component as ec, feed as feedmod, updater
from strata_cli.update_cmd import hardware_for_request

WIN = "windows-x86_64"

MIXED = [{"vendor": "10de", "device": 0x2182, "name": "NVIDIA GeForce GTX 1660 Ti", "vram_gb": 6.0},
         {"vendor": "1002", "device": 0x13C0, "name": "AMD Radeon(TM) Graphics", "vram_gb": 0.5},
         {"vendor": "1002", "device": 0x7550, "name": "AMD Radeon RX 9070 XT", "vram_gb": 15.92}]


def engine_package(version, backend, variant, archs):
    return feedmod._engine_package({
        "version": version, "platform": WIN, "backend": backend, "variant": variant, "archs": list(archs),
        "min_driver": {}, "min_runtime": None, "build_id": f"b-{version}-{variant}", "format": "tar.gz",
        "url": f"strata-engine-{version}-{WIN}-{backend}-{variant}.tar.gz", "size": 10,
        "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 20, "notes": ""})


# The Windows part of the 0.1.52 feed as the reporter's run lists it: two CUDA builds and three HIP builds.
FEED = [engine_package("0.1.52", "cuda", "cuda13", ("sm_75", "sm_86", "sm_89", "sm_120")),
        engine_package("0.1.52", "cuda", "cuda12", ("sm_75", "sm_86")),
        engine_package("0.1.52", "hip", "rx9070", ("gfx1200", "gfx1201")),
        engine_package("0.1.52", "hip", "rdna3", ("gfx1100", "gfx1101", "gfx1102", "gfx1151")),
        engine_package("0.1.52", "hip", "rdna2", ("gfx1030", "gfx1031"))]


class OnAWindowsPcWithBothMakes(unittest.TestCase):
    """Every case runs the real probes over the reporter's adapter list, with no GPU tool installed."""

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, attribute, kwargs in ((ec, "windows_display_adapters", {"return_value": MIXED}),
                                          (ec, "host_platform", {"return_value": WIN}),
                                          (updater, "host_platform", {"return_value": WIN}),
                                          (ec.shutil, "which", {"return_value": None}),
                                          (ec.os, "name", {"new": "nt"})):
            stack.enter_context(mock.patch.object(target, attribute, **kwargs))

    def detect_once(self):
        """Patch detect_hardware so a test can see how often it ran and with which environment."""
        calls, real = [], updater.detect_hardware

        def once(environ=None):
            calls.append(environ)
            return real(environ)

        mock.patch.object(updater, "detect_hardware", side_effect=once).start()
        self.addCleanup(mock.patch.stopall)
        return calls

    def test_no_ask_leaves_the_plain_probe_alone(self):
        """The default path is untouched: one probe, CUDA first, no AMD evidence, HIP refused."""
        calls = self.detect_once()
        got = hardware_for_request(None)
        self.assertEqual(calls, [None])
        self.assertEqual(got["backend"], "cuda")
        self.assertEqual(got["backend_source"], "display adapters")
        self.assertEqual(got["gfx"], [])
        self.assertEqual(got["gpu_names"], [])
        self.assertEqual(got["makes"], ["cuda", "hip"])
        best, reasons = feedmod.select_engine(FEED, got)
        self.assertEqual(best.variant, "cuda13")
        self.assertIn("0.1.52 windows-x86_64-hip-rx9070 is a hip engine, this PC needs cuda",
                      " ".join(reasons))

    def test_asking_for_hip_probes_the_amd_card(self):
        """`--backend hip`: the AMD card is in the hardware, so the search has something to choose from."""
        calls = self.detect_once()
        got = hardware_for_request("hip")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][ec.ENV_BACKEND], "hip")
        self.assertEqual(got["backend"], "hip")
        self.assertEqual(got["backend_source"], "requested")
        self.assertEqual(got["gfx"], ["gfx1201"])
        self.assertEqual(got["gpu_names"], ["AMD Radeon RX 9070 XT"])

    def test_asking_for_hip_reaches_the_package_for_the_card(self):
        """The card alone picks the rx9070 build; the other two AMD builds say their code does not cover gfx1201."""
        got = hardware_for_request("hip")
        best, reasons = feedmod.select_engine(FEED, got)
        self.assertEqual(best.variant, "rx9070")
        self.assertIn("0.1.52 windows-x86_64-hip-rdna3 has code for gfx1100, gfx1101, gfx1102, gfx1151, this PC's GPU is gfx1201",
                      " ".join(reasons))
        self.assertNotIn("the card was not recognised", " ".join(reasons))

    def test_a_known_card_beats_a_build_asked_for_by_name(self):
        """With the card known, its arch decides; the asked build is only the tie-breaker when the arch is
        unknown (#3486's rule: the newest build of the wrong card family must never win)."""
        got = hardware_for_request("hip-rdna3")
        self.assertEqual(got["variant_wanted"], "rdna3")
        self.assertEqual(feedmod.select_engine(FEED, got)[0].variant, "rx9070")

    def test_an_ask_that_is_not_a_make_does_not_re_probe(self):
        """A name outside the makes keeps the old behaviour: the backend is set as it was, one probe."""
        calls = self.detect_once()
        got = hardware_for_request("vulkan")
        self.assertEqual(calls, [None])
        self.assertEqual(got["backend"], "vulkan")
        self.assertEqual(got["backend_source"], "requested")


if __name__ == "__main__":
    unittest.main()
