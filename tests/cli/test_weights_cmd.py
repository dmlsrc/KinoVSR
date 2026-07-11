"""The kinovsr weights subcommand: listing, verification, exit codes."""

import pytest

from kinovsr.cli.main import main
from kinovsr.cli.weights_cmd import run_weights_command

pytestmark = pytest.mark.unit


def test_list_all_owners_exits_zero():
    assert run_weights_command(["list"]) == 0


def test_verify_this_checkout_passes():
    # spynet is bundled (present + artifact hash recorded); the external
    # families report their install state without failing the run.
    assert run_weights_command(["verify"]) == 0


def test_verify_single_owner():
    assert run_weights_command(["verify", "spynet"]) == 0


def test_unknown_owner_exits_two():
    assert run_weights_command(["verify", "nosuchfamily"]) == 2


def test_main_routes_the_weights_subcommand():
    assert main(["weights", "list"]) == 0


def test_factory_profiles_match_their_manifests():
    from kinovsr.modeling.weights import load_registered
    from kinovsr.processors.bsvd.factory import FACTORY as bsvd_factory
    from kinovsr.processors.capabilities import Capability
    from kinovsr.processors.realplksr.factory import FACTORY as realplksr_factory

    bsvd_profiles = bsvd_factory.capabilities[Capability.DENOISE].profiles
    assert set(bsvd_profiles) == set(load_registered("bsvd").profiles)

    plksr_profiles = realplksr_factory.capabilities[
        Capability.UPSCALE].profiles
    assert set(plksr_profiles) == set(load_registered("realplksr").profiles)
