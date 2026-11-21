"""Static checks of the V3 master notebook and the Windows ASR scripts (nothing is executed)."""
import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER.ipynb"


def _cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb["cells"]


def _src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def test_notebook_code_cells_compile_and_have_no_outputs():
    for c in _cells():
        if c["cell_type"] == "code":
            compile(_src(c), c.get("id", "?"), "exec")
            assert not c.get("outputs"), "notebook must be committed without outputs"


def test_notebook_uses_only_existing_helpers():
    from wildspeech.v3 import notebook as nb

    code = "\n".join(_src(c) for c in _cells() if c["cell_type"] == "code")
    used = set(re.findall(r"\bnb\.([a-zA-Z_]+)\(", code))
    missing = [u for u in used if not hasattr(nb, u)]
    assert not missing, missing
    for needed in ("eligibility_report", "installed_checkpoints", "execution_state", "transcripts",
                   "phoneme_feature_summary", "output_gate", "p24_outcome", "environment", "coverage_figure"):
        assert needed in used, needed


def test_notebook_is_gated_and_never_downloads():
    code = "\n".join(_src(c) for c in _cells() if c["cell_type"] == "code")
    assert "RUN_PIPELINE = False" in code
    for call in re.findall(r"nb\.run_stages\([^)]*\)", code):
        assert "enabled=RUN_PIPELINE" in call, call
    for bad in ("snapshot_download", "huggingface-cli download", "--execute", "subprocess", "os.system"):
        assert bad not in code, bad


def test_notebook_covers_every_stage_and_p24_subsections():
    md = "\n".join(_src(c) for c in _cells() if c["cell_type"] == "markdown")
    for s in ("P23", "P24", "P25", "P26", "P27", "P28", "P29"):
        assert f"## {s}" in md, s
    for sub in ("P24.1", "P24.2", "P24.3", "P24.4", "P24.5", "P24.6", "P24.7", "P24.8"):
        assert sub in md, sub


def test_python_scripts_parse_and_download_requires_execute():
    for name in ("v3_download_asr_models.py", "v3_verify_asr.py"):
        ast.parse((REPO / "scripts" / name).read_text(encoding="utf-8"))
    src = (REPO / "scripts" / "v3_download_asr_models.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # snapshot_download is only reachable after the `if not a.execute: continue` guard
    i_guard = src.index("if not a.execute:")
    assert src.index("snapshot_download(repo_id") > i_guard
    assert "allow_unverified_repo" in src
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)


def test_powershell_scripts_target_mitsui_and_are_opt_in():
    for name in ("v3_install_asr_env.ps1", "v3_download_asr_models.ps1", "v3_verify_asr.ps1"):
        t = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert '$CondaEnv = "MITSUI"' in t, name
        assert "conda run" in t, name
    dl = (REPO / "scripts" / "v3_download_asr_models.ps1").read_text(encoding="utf-8")
    assert "[switch]$Execute" in dl and 'if ($Execute) { $argsList += "--execute" }' in dl
    vf = (REPO / "scripts" / "v3_verify_asr.ps1").read_text(encoding="utf-8")
    assert '$env:HF_HUB_OFFLINE = "1"' in vf
    inst = (REPO / "scripts" / "v3_install_asr_env.ps1").read_text(encoding="utf-8")
    assert "[switch]$InstallTorch" in inst and "[switch]$CheckOnly" in inst
    assert "snapshot_download" not in inst and "huggingface-cli download" not in inst


def test_config_defaults_are_safe():
    import yaml

    v3 = yaml.safe_load((REPO / "configs" / "v3.yaml").read_text(encoding="utf-8"))["v3"]["asr"]
    assert v3["enabled"] is False and v3["accept_fixture_outputs"] is False
    assert v3["models"][:2] == ["pasketti_phonetic", "whisper_small_en"]
    reg = yaml.safe_load((REPO / "configs" / "v3_asr_models.yaml").read_text(encoding="utf-8"))
    for k, m in reg["models"].items():
        for flag in ("license_verified", "provenance_verified", "overlap_checked", "competition_eligible"):
            assert flag not in m, f"{k}: approvals must not live in the registry ({flag})"
    assert (REPO / "configs" / "archive" / "v3_asr_models.schema1.yaml").exists()
