"""Shared operator build and execution utilities."""
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

CSR_MAP = "ame-local-810-811-812-v1"

ENTRY = "m-mode-single-hart-ms-enabled-unowned-v1"

def load(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def config(path):
    c = load(path)
    require(c["schema"] == 1 and c["csr_map"] == CSR_MAP, "unknown schema/CSR map")
    require(4 <= c["max_n"] <= 128 and c["max_n"] & (c["max_n"] - 1) == 0,
            "max_n must be a power of two in [4,128]")
    require(0 <= c["seed"] <= 0xffffffff, "seed must fit u32")
    require(set(c["xlens"]) <= {32, 64} and c["xlens"], "invalid XLENs")
    require(c["optimizations"] and set(c["optimizations"]) <= {"O0", "O2"}, "unsupported optimization")
    require(c["languages"] and set(c["languages"]) <= {"c", "c++"}, "unsupported language")
    require(c["profiles"], "empty profiles")
    for p in c["profiles"]:
        require(p["uds"] in (8, 16, 32), "first batch requires complete u32 squares")
        require(p["mregs"] in (16, 32) and p["accregs"] in (1, 2, 4, 8, 16),
                "invalid register profile")
    require(1 <= c["timeout_seconds"] <= 600, "invalid timeout")
    require(256 <= c["address_space_mib"] <= 4096, "invalid memory cap")
    return c

def tool_paths(c):
    root = Path(c["toolchain_root"])
    gcc = root / c["gcc_build"]
    binutils = root / c["binutils_build"]
    return dict(cc=gcc / "xgcc", cxx=gcc / "xg++", cc1=gcc / "cc1",
                cc1plus=gcc / "cc1plus", assembler=binutils / "gas/as-new",
                linker=binutils / "ld/ld-new", objdump=binutils / "binutils/objdump",
                readelf=binutils / "binutils/readelf", nm=binutils / "binutils/nm-new")

def command(argv, directory, tag, c, records):
    argv = [str(a) for a in argv]
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (c["address_space_mib"] * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    start = time.time()
    log = directory / (tag + ".log")
    with log.open("wb") as output:
        proc = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                                cwd=ROOT, start_new_session=True, preexec_fn=limits)
        try:
            status = proc.wait(timeout=c["timeout_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = 124
    records.append(dict(argv=argv, exit_status=status, log=str(log),
                        start=start, elapsed=time.time() - start))
    save(directory / "commands.json", records)
    require(status == 0, f"{tag} failed ({status}); see {log}")
    return log.read_text(errors="replace")

def tool_identity(paths):
    result = {}
    for path in paths:
        st = Path(path).stat()
        result[str(path)] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    return result

def tool_snapshot(paths):
    paths = list(paths)
    before = tool_identity(paths)
    hashes = {str(p): sha(p) for p in paths}
    require(before == tool_identity(paths), "tools changed during snapshot")
    return dict(identity=before, hashes=hashes)

def check_result_labels(text, expected):
    actual = {}
    labels = {"PASS", "FAIL", "UNRESOLVED", "UNSUPPORTED", "UNTESTED", "ERROR"}
    for line in text.splitlines():
        status, sep, label = line.partition(": ")
        if sep and status in labels:
            actual.setdefault(label, []).append(status)
    return (set(actual) == set(expected) and all(
        len(actual[label]) == 1 and actual[label][0] in allowed
        for label, allowed in expected.items()))

def validate_mailbox(blob, build_report, n):
    require(len(blob) == 128, "mailbox must be exactly 128 bytes")
    w = struct.unpack("<32I", blob)
    r, c = build_report["row"], build_report["config"]
    expected = {0: 0x414d4554, 1: 1, 3: r["case"]["id"], 4: r["xlen"],
                5: n, 6: r["profile"]["uds"], 7: r["profile"]["mregs"],
                8: r["profile"]["accregs"], 9: c["seed"]}
    require(w[2] == 2, f"guest did not PASS: status={w[2]} error={w[11]} "
            f"index={w[12]:#x} expected={w[13]:#x} actual={w[14]:#x} "
            f"mcause={(w[19] << 32) | w[16]:#x} pc={(w[20] << 32) | w[17]:#x}")
    require(all(w[k] == v for k, v in expected.items()), "mailbox identity/config mismatch")
    require(w[10] >= 6 + 4 * (c["max_n"]**2 + 32), "missing guest checks")
    require(not any(w[11:15]) and not any(w[16:]), "PASS has error/trap/reserved words")
    require(w[15] & 1 == 1, "missing successful initial acquisition")
    return dict(status="EXEC_PASS", checks=w[10], n=n)

def execute(report, platform, n, directory):
    directory.mkdir(parents=True, exist_ok=False)
    require(sha(report["elf"]) == report["elf_sha256"], "ELF changed after build")
    require(platform["schema"] == 1 and platform["csr_map"] == CSR_MAP,
            "platform CSR/schema mismatch")
    require(platform["entry_contract"] == ENTRY, "platform entry contract mismatch")
    if not platform.get("command"):
        result = dict(status="NOT_RUN", reason="execution adapter not configured")
        save(directory / "result.json", result)
        return 77
    row = report["row"]
    require(platform["xlen"] == row["xlen"] and all(
        platform[k] == row["profile"][k] for k in ("uds", "mregs", "accregs")),
        "build/platform profile mismatch")
    require(n in platform["n_values"] and 4 <= n <= report["config"]["max_n"]
            and n & (n - 1) == 0, "invalid runtime N")
    caps = platform.get("capabilities")
    require(isinstance(caps, list), "explicit platform capability list is required")
    missing = set(row["case"]["capabilities"]) - set(caps)
    if missing:
        save(directory / "result.json", dict(status="UNSUPPORTED", missing=sorted(missing)))
        return 78
    require(platform.get("revision"), "backend revision is required")
    argv = platform["command"]
    require(isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv),
            "adapter command must be argv, not shell text")
    require(Path(argv[0]).is_absolute(), "adapter executable must be absolute")
    nonce = secrets.token_hex(16)
    expected = dict(xlen=row["xlen"], n=n, **row["profile"], csr_map=CSR_MAP,
                    entry_contract=ENTRY)
    request = dict(schema=1, nonce=nonce, elf=report["elf"], elf_sha256=report["elf_sha256"],
                   symbols=report["symbols"], mailbox_bytes=128,
                   expected=expected, backend=platform["backend"], revision=platform["revision"])
    save(directory / "request.json", request)
    save(directory / "platform.json", platform)
    records = []
    limits = dict(report["config"], timeout_seconds=platform["timeout_seconds"])
    command(argv + ["--request", directory / "request.json", "--response", directory / "response.json"],
            directory, "adapter", limits, records)
    response = load(directory / "response.json")
    result = validate_response(response, request, report, n)
    result.update(elf_sha256=report["elf_sha256"], backend=platform["backend"],
                  revision=platform["revision"], adapter_sha256=sha(argv[0]))
    save(directory / "result.json", result)
    return 0

def validate_response(response, request, report, n):
    require(response["schema"] == 1 and response["nonce"] == request["nonce"], "stale adapter response")
    require(response["elf_sha256"] == report["elf_sha256"], "executed ELF hash mismatch")
    require(response["observed"] == request["expected"], "observed platform differs from request")
    require(response["backend"] == request["backend"] and response["revision"] == request["revision"],
            "executed backend revision mismatch")
    require(response["stop_pc"] == report["symbols"]["ame_test_done"], "did not stop at completion")
    return validate_mailbox(bytes.fromhex(response["mailbox_hex"]), report, n)
