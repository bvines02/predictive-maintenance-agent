"""Stage 14h - SYNTHETIC asset-context profiles for the decision-engine demo.

These are invented for the learning exercise. NASA C-MAPSS has no asset
criticality, redundancy or spares lead time. In a real deployment each field
would be read from a system of record (asset register / FMEA, P&ID or
knowledge graph, CMMS / ERP) - never hard-coded here.
"""

from src.decision_engine import AssetContext, Confidence, Criticality, Redundancy

SYNTHETIC_PROFILES = {
    "P1 non-critical redundant": AssetContext(
        criticality=Criticality.LOW,
        redundancy=Redundancy.FULL,
        maintenance_lead_time_cycles=5,
        confidence=Confidence.HIGH,
    ),
    "P2 important asset": AssetContext(
        criticality=Criticality.HIGH,
        redundancy=Redundancy.PARTIAL,
        maintenance_lead_time_cycles=15,
        confidence=Confidence.HIGH,
    ),
    "P3 critical single point": AssetContext(
        criticality=Criticality.CRITICAL,
        redundancy=Redundancy.NONE,
        maintenance_lead_time_cycles=25,
        confidence=Confidence.HIGH,
    ),
}
