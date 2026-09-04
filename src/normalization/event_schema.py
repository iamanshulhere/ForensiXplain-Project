from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ForensicEvent:
    case_id: str
    event_id: str

    timestamp: Optional[str] = None
    timestamp_end: Optional[str] = None
    timestamp_confidence: str = "unknown"

    source: str = "memory"
    artifact_type: str = ""
    event_type: str = ""
    action: str = ""

    user: Optional[str] = None
    host: Optional[str] = None

    process: Optional[str] = None
    parent_process: Optional[str] = None
    process_id: Optional[int] = None
    parent_process_id: Optional[int] = None

    command_line: Optional[str] = None

    file: Optional[str] = None
    file_path: Optional[str] = None
    file_hash: Optional[str] = None

    device: Optional[str] = None

    ip_src: Optional[str] = None
    ip_dst: Optional[str] = None
    port_src: Optional[int] = None
    port_dst: Optional[int] = None
    protocol: Optional[str] = None

    parent_event_id: Optional[str] = None
    related_event_id: Optional[str] = None
    relationship: Optional[str] = None

    evidence_id: Optional[str] = None
    provenance: Optional[str] = None

    def to_dict(self):
        return asdict(self)