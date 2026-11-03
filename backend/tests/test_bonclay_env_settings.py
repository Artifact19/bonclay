"""The ./bonclay start script reads the settings backend/.env documents (BONCLAY_TUNNEL, PHONE_PUBLIC_PORT,
BONCLAY_KEEP_AWAKE, PUBLIC_BASE_URL) from the shell first, else from backend/.env, else a default.

Each test copies the script into a temporary directory next to a temporary backend/.env (the script finds backend/ from its
own location), so no real .env is read and nothing touches the network. The script has three hidden subcommands for this:
`_setting KEY [default]`, `_tunnel_kind` and `_start_tunnel`. A stub `ngrok` / `cloudflared` on PATH records whether the
script tried to start a real tunnel."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "bonclay"
SETTING_KEYS = ("BONCLAY_TUNNEL", "PHONE_PUBLIC_PORT", "BONCLAY_KEEP_AWAKE", "PUBLIC_BASE_URL")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


class Home:
    def __init__(self, root: Path):
        self.root = root
        (root / "backend").mkdir()
        shutil.copy(SCRIPT, root / "bonclay")
        self.bin = root / "bin"
        self.bin.mkdir()
        self.marker = root / "tunnel-started"

    def env_file(self, text: str):
        (self.root / "backend" / ".env").write_text(text, encoding="utf-8")

    def stub(self, name: str):
        p = self.bin / name
        p.write_text(f"#!/bin/sh\necho {name} >> '{self.marker}'\nsleep 0\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC)

    def run(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if k not in SETTING_KEYS}
        e["PATH"] = f"{self.bin}{os.pathsep}{e['PATH']}"
        e.update(env or {})
        r = subprocess.run(["bash", str(self.root / "bonclay"), *args], capture_output=True, text=True, env=e,
                           cwd=self.root, timeout=60)
        return r.stdout.strip()


@pytest.fixture
def home(tmp_path):
    return Home(tmp_path)


def test_shell_environment_wins_over_env_file(home):
    home.env_file("PHONE_PUBLIC_PORT=9001\nBONCLAY_TUNNEL=off\n")
    assert home.run("_setting", "PHONE_PUBLIC_PORT", "8010", env={"PHONE_PUBLIC_PORT": "9500"}) == "9500"
    assert home.run("_tunnel_kind", env={"BONCLAY_TUNNEL": "cloudflared"}) == "cloudflared"


def test_env_file_used_when_the_shell_is_unset(home):
    home.env_file("PHONE_PUBLIC_PORT=9001\nBONCLAY_TUNNEL=off\n")
    assert home.run("_setting", "PHONE_PUBLIC_PORT", "8010") == "9001"
    assert home.run("_tunnel_kind") == "off"


def test_default_when_neither_is_set(home):
    home.env_file("SOMETHING_ELSE=1\n")
    assert home.run("_setting", "PHONE_PUBLIC_PORT", "8010") == "8010"
    assert home.run("_setting", "BONCLAY_KEEP_AWAKE", "on") == "on"
    (home.root / "backend" / ".env").unlink()  # no file at all is fine
    assert home.run("_setting", "PHONE_PUBLIC_PORT", "8010") == "8010"
    assert home.run("_setting", "PUBLIC_BASE_URL") == ""


@pytest.mark.parametrize("text, expected", [
    ("# PHONE_PUBLIC_PORT=1111\nPHONE_PUBLIC_PORT=9001\n", "9001"),                    # comment lines are ignored
    ("\n\n   \nPHONE_PUBLIC_PORT=9001\n", "9001"),                                     # blank lines
    ('PHONE_PUBLIC_PORT="9001"\n', "9001"),                                            # double quotes
    ("PHONE_PUBLIC_PORT='9001'\n", "9001"),                                            # single quotes
    ("PHONE_PUBLIC_PORT=9001   # the phone port\n", "9001"),                           # trailing comment
    ('PHONE_PUBLIC_PORT="9001"  # quoted, then a comment\n', "9001"),
    ("PHONE_PUBLIC_PORT=1111\nPHONE_PUBLIC_PORT=9001\n", "9001"),                      # last assignment wins
    ("  export PHONE_PUBLIC_PORT = 9001\r\n", "9001"),                                 # export, spaces, CRLF
    ("PHONE_PUBLIC_PORT=9001\nPHONE_PUBLIC_PORT=\n", "8010"),                          # a later empty value means the default
    ("XPHONE_PUBLIC_PORT=1\nPHONE_PUBLIC_PORT_X=2\n", "8010"),                         # only the exact key
    ("PHONE_PUBLIC_PORT=9001", "9001"),                                                # no trailing newline
])
def test_env_file_parsing(home, text, expected):
    home.env_file(text)
    assert home.run("_setting", "PHONE_PUBLIC_PORT", "8010") == expected


def test_url_keeps_a_hash_that_is_not_a_comment(home):
    home.env_file("PUBLIC_BASE_URL=https://phone.example.in/path#frag  # mine\n")
    assert home.run("_setting", "PUBLIC_BASE_URL") == "https://phone.example.in/path#frag"


def test_tunnel_off_in_env_file_starts_no_tunnel(home):
    home.stub("ngrok")
    home.stub("cloudflared")
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off  # my own named tunnel\nPUBLIC_BASE_URL=https://phone.example.in\n")
    out = home.run("_start_tunnel")
    assert not home.marker.exists(), "ngrok or cloudflared was started although BONCLAY_TUNNEL=off"
    assert "Tunnel  https://phone.example.in  (your own tunnel; BONCLAY_TUNNEL=off)" in out


def test_tunnel_off_without_public_base_url_says_what_is_missing(home):
    home.stub("ngrok")
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off\nPUBLIC_BASE_URL=\n")
    out = home.run("_start_tunnel")
    assert not home.marker.exists()
    assert "PUBLIC_BASE_URL" in out and "phone calls need" in out


def test_tunnel_off_takes_public_base_url_from_the_shell_too(home):
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off\n")
    out = home.run("_start_tunnel", env={"PUBLIC_BASE_URL": "https://shell.example.in"})
    assert "https://shell.example.in" in out


def test_status_shows_the_own_tunnel_instead_of_not_running(home):
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off\nPUBLIC_BASE_URL=https://phone.example.in\n")
    out = home.run("status")
    assert "https://phone.example.in  (your own tunnel; BONCLAY_TUNNEL=off)" in out
    assert "Tunnel not running" not in out
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off\n")
    out = home.run("status")
    assert "phone calls need PUBLIC_BASE_URL" in out and "Tunnel not running" not in out


def test_auto_still_picks_an_installed_tunnel_and_off_in_the_shell_wins(home):
    home.stub("cloudflared")
    home.env_file("EXOTEL_API_KEY=test-key\n")
    assert home.run("_tunnel_kind") in ("ngrok", "cloudflared")  # auto: whichever is installed (ngrok first)
    home.env_file("BONCLAY_TUNNEL=ngrok\n")
    assert home.run("_tunnel_kind") == "ngrok"
    assert home.run("_tunnel_kind", env={"BONCLAY_TUNNEL": "off"}) == "off"


def test_doctor_tunnel_check_is_skippable_and_never_needed_for_other_kinds(home):
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off\nPUBLIC_BASE_URL=https://phone.example.in\n")
    out = home.run("doctor", env={"BONCLAY_DOCTOR_SKIP_TUNNEL_CHECK": "1"})
    assert "Phone tunnel" in out and "reachability check skipped" in out
    home.env_file("EXOTEL_API_KEY=test-key\nBONCLAY_TUNNEL=off\n")
    assert "PUBLIC_BASE_URL is empty" in home.run("doctor", env={"BONCLAY_DOCTOR_SKIP_TUNNEL_CHECK": "1"})
