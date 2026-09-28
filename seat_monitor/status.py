# SPDX-License-Identifier: AGPL-3.0-only
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


def snapshot(states, scores, now, health="ok", ttl_seconds=5, observed_at=None):
    # Freshness starts at capture time, not after a potentially slow inference.
    wall = observed_at or datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "observed_at": wall.isoformat(),
        "valid_until": (wall + timedelta(seconds=ttl_seconds)).isoformat(),
        "health": health,
        "meaning": "person_presence_only",
        "seats": [
            {
                "seat_id": seat_id,
                "state": state.state,
                "has_person": {"OCCUPIED": True, "EMPTY": False, "UNKNOWN": None}[state.state],
                "current_person_confidence": round(scores.get(seat_id, 0.0), 4),
                "pending_state": state.candidate,
                "state_duration_seconds": round(max(0.0, now - state.state_since), 2)
                if state.state_since is not None else 0.0,
            }
            for seat_id, state in states.items()
        ],
    }


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
