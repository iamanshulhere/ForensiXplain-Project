from pathlib import Path
import pandas as pd

from event_schema import ForensicEvent


CASE_ID = "M57-Jean"
SOURCE = "memory"

INPUT_DIR = Path("data/extracted/M57-Jean/memory")
OUTPUT_DIR = Path("data/normalized/M57-Jean")


def clean_value(value):
    """Convert empty/invalid values into None."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"", "-", "N/A", "nan", "None"}:
        return None

    return value


def normalize_timestamp(value):
    """Convert a forensic timestamp to ISO-8601 UTC."""
    value = clean_value(value)

    if value is None:
        return None

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(timestamp):
        return None

    return timestamp.isoformat()


def load_csv(filename):
    path = INPUT_DIR / filename

    if not path.exists():
        print(f"[WARNING] Missing: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-16")
        print(f"[OK] Loaded {filename}: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"[ERROR] Could not read {filename}: {exc}")
        return pd.DataFrame()


def normalize_pslist(df, events):
    """
    Convert process-list observations into process_start events.
    """

    if df.empty:
        return

    for index, row in df.iterrows():

        pid = clean_value(row.get("PID"))
        ppid = clean_value(row.get("PPID"))
        process = clean_value(row.get("ImageFileName"))
        timestamp = normalize_timestamp(row.get("CreateTime"))

        evidence_id = f"{CASE_ID}-MEM-PSLIST-{index}"

        event_id = f"EVT-{CASE_ID}-PROCESS-{index}"

        event = ForensicEvent(
            case_id=CASE_ID,
            event_id=event_id,
            timestamp=timestamp,
            timestamp_confidence=(
                "observed" if timestamp else "unknown"
            ),
            source=SOURCE,
            artifact_type="pslist",
            event_type="process",
            action="process_start",

            process=process,
            process_id=int(pid) if pid else None,
            parent_process_id=int(ppid) if ppid else None,

            evidence_id=evidence_id,

            provenance=(
                "Volatility 3 windows.pslist "
                "from M57-Jean memory image"
            ),
        )

        events.append(event)


def normalize_pstree(df, events):
    """
    Convert parent-child process relationships
    into relationship events.
    """

    if df.empty:
        return

    for index, row in df.iterrows():

        pid = clean_value(row.get("PID"))
        ppid = clean_value(row.get("PPID"))

        process = clean_value(row.get("ImageFileName"))

        timestamp = normalize_timestamp(
            row.get("CreateTime")
        )

        if not pid or not ppid:
            continue

        evidence_id = f"{CASE_ID}-MEM-PSTREE-{index}"

        event_id = f"EVT-{CASE_ID}-PARENT-{index}"

        event = ForensicEvent(
            case_id=CASE_ID,
            event_id=event_id,

            timestamp=timestamp,

            timestamp_confidence=(
                "observed" if timestamp else "unknown"
            ),

            source=SOURCE,
            artifact_type="pstree",
            event_type="process_relationship",
            action="parent_of",

            process=process,
            process_id=int(pid),
            parent_process_id=int(ppid),

            relationship="parent_of",

            evidence_id=evidence_id,

            provenance=(
                "Volatility 3 windows.pstree "
                "from M57-Jean memory image"
            ),
        )

        events.append(event)


def normalize_cmdline(df, events):
    """
    Convert command-line observations.
    No timestamp is inferred here.
    """

    if df.empty:
        return

    for index, row in df.iterrows():

        pid = clean_value(row.get("PID"))
        process = clean_value(row.get("Process"))
        command_line = clean_value(row.get("Args"))

        if not process or not command_line:
            continue

        evidence_id = f"{CASE_ID}-MEM-CMDLINE-{index}"

        event_id = f"EVT-{CASE_ID}-CMD-{index}"

        event = ForensicEvent(
            case_id=CASE_ID,
            event_id=event_id,

            timestamp=None,
            timestamp_confidence="unknown",

            source=SOURCE,
            artifact_type="cmdline",
            event_type="command_line",
            action="observed",

            process=process,
            process_id=int(pid) if pid else None,

            command_line=command_line,

            evidence_id=evidence_id,

            provenance=(
                "Volatility 3 windows.cmdline "
                "from M57-Jean memory image"
            ),
        )

        events.append(event)


def normalize_dlllist(df, events):
    """
    Convert loaded-DLL observations.
    """

    if df.empty:
        return

    for index, row in df.iterrows():

        pid = clean_value(row.get("PID"))
        process = clean_value(row.get("Process"))
        dll_name = clean_value(row.get("Name"))
        dll_path = clean_value(row.get("Path"))

        if not process:
            continue

        evidence_id = f"{CASE_ID}-MEM-DLL-{index}"

        event_id = f"EVT-{CASE_ID}-DLL-{index}"

        event = ForensicEvent(
            case_id=CASE_ID,
            event_id=event_id,

            timestamp=None,
            timestamp_confidence="unknown",

            source=SOURCE,
            artifact_type="dlllist",
            event_type="module",
            action="loaded",

            process=process,
            process_id=int(pid) if pid else None,

            file=dll_name,
            file_path=dll_path,

            evidence_id=evidence_id,

            provenance=(
                "Volatility 3 windows.dlllist "
                "from M57-Jean memory image"
            ),
        )

        events.append(event)


def normalize_malfind(df, events):
    """
    Convert malfind results into memory-region observations.

    These are observations, NOT confirmed malware labels.
    """

    if df.empty:
        return

    for index, row in df.iterrows():

        pid = clean_value(row.get("PID"))
        process = clean_value(row.get("Process"))
        start_vpn = clean_value(row.get("Start VPN"))
        end_vpn = clean_value(row.get("End VPN"))
        protection = clean_value(row.get("Protection"))
        tag = clean_value(row.get("Tag"))
        notes = clean_value(row.get("Notes"))

        if not process:
            continue

        evidence_id = f"{CASE_ID}-MEM-MALFIND-{index}"

        event_id = f"EVT-{CASE_ID}-MEMORY-{index}"

        event = ForensicEvent(
            case_id=CASE_ID,
            event_id=event_id,

            timestamp=None,
            timestamp_confidence="unknown",

            source=SOURCE,
            artifact_type="malfind",
            event_type="memory_region",
            action="suspicious_region_observed",

            process=process,
            process_id=int(pid) if pid else None,

            file_path=(
                f"{start_vpn} - {end_vpn}"
                if start_vpn or end_vpn
                else None
            ),

            relationship=protection,

            evidence_id=evidence_id,

            provenance=(
                "Volatility 3 windows.malware.malfind "
                "observation from M57-Jean memory image"
            ),
        )

        events.append(event)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events = []

    print("\n=== ForensiXplain Memory Normalization ===\n")

    pslist = load_csv("pslist.csv")
    pstree = load_csv("pstree.csv")
    cmdline = load_csv("cmdline.csv")
    dlllist = load_csv("dlllist.csv")
    malfind = load_csv("malfind.csv")

    normalize_pslist(pslist, events)
    normalize_pstree(pstree, events)
    normalize_cmdline(cmdline, events)
    normalize_dlllist(dlllist, events)
    normalize_malfind(malfind, events)

    records = [event.to_dict() for event in events]

    output = pd.DataFrame(records)

    output_path = OUTPUT_DIR / "events.csv"

    output.to_csv(output_path, index=False)

    print("\n=== Normalization Complete ===")
    print(f"Total events: {len(output)}")
    print(f"Output: {output_path}")

    if not output.empty:
        print("\nEvents by artifact:")
        print(output["artifact_type"].value_counts())

        print("\nEvents by type:")
        print(output["event_type"].value_counts())


if __name__ == "__main__":
    main()