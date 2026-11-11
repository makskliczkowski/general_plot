r"""Record local writer information without probing accelerators or private identifiers.

Producer information belongs to the caller. A writer record describes where an
artifact was saved, not necessarily where its input arrays were computed.

-------------------------------------------
Author: Maksymilian Kliczkowski
File: general_plot/data/provenance.py
-------------------------------------------
"""
from __future__ import annotations

import hashlib
import platform
from typing import Any, Iterable, Mapping, Optional


def source_info(*, devices: Iterable[Mapping[str, Any]] = ()) -> dict:
    """Describe the local host and explicitly supplied device placements.

    Parameters
    ----------
    devices : iterable of mappings
        Caller-known device descriptors, such as platform, kind and logical id.
        No backend is imported or initialized. An empty list means unknown.

    Returns
    -------
    dict
        OS, architecture, a truncated SHA-256 hostname identifier, and devices.
        No raw hostname, user name, IP, MAC, serial, paths or environment is stored.
        The identifier distinguishes hostnames, not persistent physical hardware.
    """
    hostname = platform.node()
    return {"host_id"   : hashlib.sha256(hostname.encode()).hexdigest()[:16] if hostname else None,
            "system"    : platform.system(), "machine": platform.machine(),
            "devices"   : [dict(device) for device in devices]}


def with_provenance(metadata: Optional[Mapping[str, Any]] = None, *, writer: Optional[Mapping[str, Any]] = None) -> dict:
    """Copy metadata and add writer provenance only when it is absent.

    Parameters
    ----------
    metadata : mapping, optional
        JSON metadata. Existing provenance, including producer information and
        explicit unknown values, is preserved. The input mapping is not mutated.
    writer : mapping, optional
        Explicit writer record. Otherwise source_info collects local host facts.

    Returns
    -------
    dict
        Metadata with provenance.writer. No producer origin is inferred.
    """
    result = dict(metadata or {})
    provenance = dict(result.get("provenance", {}))
    if "writer" not in provenance:
        provenance["writer"] = dict(writer) if writer is not None else source_info()
    result["provenance"] = provenance
    return result


def source_fields(metadata: Mapping[str, Any]) -> dict:
    """Extract source columns for catalog tables and grouping without reading data.

    Parameters
    ----------
    metadata : mapping
        Run metadata containing optional provenance.writer and provenance.producer.

    Returns
    -------
    dict
        Writer/producer host IDs and device lists plus writer OS and architecture.
        Missing provenance stays None. Paths are never used to guess devices.
    """
    provenance  = metadata.get("provenance") or {}
    writer      = provenance.get("writer") or {}
    producer    = provenance.get("producer") or {}
    return {"writer_host_id"    : writer.get("host_id"), 
            "producer_host_id"  : producer.get("host_id"),
            "writer_label"      : writer.get("label"), 
            "producer_label"    : producer.get("label"),
            "writer_system"     : writer.get("system"), 
            "writer_machine"    : writer.get("machine"),
            "writer_devices"    : writer.get("devices"), 
            "producer_devices"  : producer.get("devices")}


# -------------------------------------------
#! EOF
# -------------------------------------------
