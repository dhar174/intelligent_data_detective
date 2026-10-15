"""Opt-in real-kernel evidence; CI provisions Jupyter and runs this separately."""

import os

import pytest

from run_notebook_live import configure_skip_notebook_installs, probe_kernel_scientific_stack


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("IDD_RUN_KERNEL_SMOKE") != "1",
        reason="Real Jupyter smoke is a separate provisioned gate; set IDD_RUN_KERNEL_SMOKE=1 and IDD_KERNEL_NAME.",
    ),
]


@pytest.mark.parametrize("flag", [None, " TRUE ", "0"])
def test_selected_kernel_scientific_imports_and_install_flag(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("IDD_SKIP_NOTEBOOK_INSTALLS", raising=False)
    else:
        monkeypatch.setenv("IDD_SKIP_NOTEBOOK_INSTALLS", flag)
    configure_skip_notebook_installs()
    kernel = os.environ["IDD_KERNEL_NAME"]
    assert probe_kernel_scientific_stack(kernel)
