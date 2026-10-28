"""Replay one recorded call as a sequence of speakers.

The lines come from the response cache. Nothing here calls a model, and the
agents never import this module. Wellness and productivity stay silent because
the runner that produced these logs never called them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ato.faults import LIBRARY, inject_state_fault
from ato.schema import WorldState
from viz.board import FIXTURES

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRS = (ROOT / ".cache" / "gate2", ROOT / ".cache" / "gate-haiku")


def witness_call(
    scenario_id: str,
    *,
    model: str,
    condition: str,
    seed: int = 1,
) -> dict[str, Any]:
    run = _run(scenario_id, model=model, condition=condition, seed=seed)
    if run is None:
        return {
            "scenario_id": scenario_id,
            "model": model,
            "condition": condition,
            "seed": seed,
            "action": None,
            "beats": [
                _beat(
                    "observer",
                    "No recorded call matches that scenario, model, and condition.",
                )
            ],
        }
    state = _expected_state(scenario_id, run["fault_id"])
    action = next(
        (item for item in state.candidate_actions if item.id == run["chosen_action"]),
        None,
    )
    fiduciary = _final_call(_calls(state, model, seed, "fiduciary", condition, run))
    blame = _final_call(_calls(state, model, seed, "blame", condition, run))
    beats = _script(run, action.desc if action else run["chosen_action"], fiduciary, blame)
    return {
        "scenario_id": scenario_id,
        "model": model,
        "condition": condition,
        "seed": seed,
        "action": None if action is None else {"id": action.id, "desc": action.desc},
        "beats": beats,
    }


def _beat(speaker: str, text: str, kind: str = "say") -> dict[str, str]:
    return {"speaker": speaker, "kind": kind, "text": text}


def _script(
    run: dict[str, Any],
    action_desc: str | None,
    fiduciary: dict[str, Any] | None,
    blame: dict[str, Any] | None,
) -> list[dict[str, str]]:
    beats = [
        _beat(
            "observer",
            "I'm watching one call. The agents never speak to each other. "
            "Each would see only its own slice.",
        ),
        _beat("observer", "Wellness and productivity have seats. This run did not call them."),
        _beat("wellness", "Not called.", "silent"),
        _beat("productivity", "Not called.", "silent"),
        _beat(
            "observer",
            "The fiduciary is handed its slice. Its tools run before the model speaks.",
        ),
    ]
    if fiduciary is None:
        beats.append(_beat("fiduciary", "The reply is not in the local cache.", "silent"))
    else:
        beats.extend(_fiduciary_beats(fiduciary))
    chosen = action_desc or run["chosen_action"] or "nothing"
    beats.append(
        _beat(
            "observer",
            "The arbiter is not a model. It scores the verdicts on the table and commits.",
        )
    )
    beats.append(_beat("arbiter", f"I commit {chosen}."))
    beats.append(
        _beat(
            "observer",
            "The choice is frozen. A second call asks whether an input was wrong. "
            "That answer cannot change the choice.",
        )
    )
    if blame is None:
        beats.append(_beat("arbiter", "The blame reply is not in the local cache.", "silent"))
    else:
        beats.extend(_blame_beats(blame))
    beats.append(_beat("observer", _closing(run)))
    return beats


def _fiduciary_beats(call: dict[str, Any]) -> list[dict[str, str]]:
    beats: list[dict[str, str]] = []
    user = _parse_json(_first_user(call))
    observations = user.get("observations") if isinstance(user, dict) else None
    if isinstance(observations, list) and observations:
        names = []
        for row in observations:
            if isinstance(row, dict) and isinstance(row.get("tool"), str):
                names.append(row["tool"])
        if names:
            listed = ", ".join(dict.fromkeys(names))
            beats.append(_beat("fiduciary", f"Tools already on the table: {listed}."))
    beats.extend(_spoken(call, "fiduciary"))
    return beats


def _blame_beats(call: dict[str, Any]) -> list[dict[str, str]]:
    return _spoken(call, "arbiter")


def _spoken(call: dict[str, Any], speaker: str) -> list[dict[str, str]]:
    beats: list[dict[str, str]] = []
    messages = call["messages"]
    # The opening system and user messages are the assignment, not a turn.
    started = False
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system":
            continue
        if role == "user" and not started:
            started = True
            continue
        started = True
        if role == "user" and "failed validation" in content:
            beats.append(_beat("observer", "That reply did not parse. The call gets one more try."))
            continue
        if role == "user":
            beats.append(_beat(speaker, "A tool answered.", "tool"))
            continue
        beats.extend(_from_text(content, speaker))
    beats.extend(_from_text(str(call["response"]), speaker))
    return beats


def _from_text(content: str, speaker: str) -> list[dict[str, str]]:
    payload = _parse_json(content)
    if not isinstance(payload, dict):
        if '"tool"' in content:
            return [
                _beat(speaker, "Started a tool call, but the reply was not valid JSON.", "tool")
            ]
        return []
    if "tool" in payload and "verdicts" not in payload and "agent" not in payload:
        name = payload.get("tool")
        return [_beat(speaker, f"Calling {name}.", "tool")]
    if "verdicts" in payload and isinstance(payload["verdicts"], list):
        lines = []
        for verdict in payload["verdicts"]:
            if not isinstance(verdict, dict):
                continue
            action = verdict.get("action_id", "?")
            score = verdict.get("score")
            rationale = verdict.get("rationale") or ""
            lines.append(_beat(speaker, f"{action}, score {score}. {rationale}".strip()))
        return lines
    if "field" in payload or "agent" in payload:
        field = payload.get("field")
        if not field:
            return [_beat(speaker, "I don't see a corrupted input.")]
        confidence = payload.get("confidence")
        return [_beat(speaker, f"The wrong input looks like {field}. Confidence {confidence}.")]
    return []


def _closing(run: dict[str, Any]) -> str:
    label = run["label"]
    field = run["blame_field"]
    if label == "hit":
        return f"The observer's note: the blame call named {field}."
    if label == "clear":
        return "The observer's note: the blame call abstained."
    if label == "false alarm":
        return (
            f"The observer's note: the blame call named {field} "
            "on a run with nothing detectable to find."
        )
    return "The observer's note: the blame call did not name the field."


def _run(scenario_id: str, *, model: str, condition: str, seed: int) -> dict[str, Any] | None:
    from viz.board import scenario_view

    matches = [
        row
        for row in scenario_view(scenario_id)["runs"]
        if row["model"] == model and row["condition"] == condition and row["seed"] == seed
    ]
    return matches[0] if matches else None


def _expected_state(scenario_id: str, fault_id: str | None) -> WorldState:
    state = WorldState.model_validate_json((FIXTURES / f"{scenario_id}.json").read_text())
    if not fault_id:
        return state
    spec = next(item for item in LIBRARY if item.fault_id == fault_id)
    state, _record = inject_state_fault(state, spec)
    return state


def _calls(
    state: WorldState,
    model: str,
    seed: int,
    speaker: str,
    condition: str,
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    found = []
    for directory in CACHE_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.json"):
            call = _load_call(path)
            if call is None:
                continue
            if call["model"] != model or call["seed"] != seed:
                continue
            if not _same_world(call, state, speaker, condition, run):
                continue
            found.append(call)
    return found


def _load_call(path: Path) -> dict[str, Any] | None:
    try:
        stored = json.loads(path.read_text())
        fingerprint = stored["fingerprint"]
        messages = fingerprint["messages"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not messages:
        return None
    system = str(messages[0].get("content", ""))
    if system.startswith("You are the fiduciary agent"):
        speaker = "fiduciary"
    elif system.startswith("The action is already committed"):
        speaker = "blame"
    else:
        return None
    return {
        "speaker": speaker,
        "model": fingerprint.get("model"),
        "seed": fingerprint.get("seed"),
        "messages": messages,
        "response": stored.get("response", {}).get("text", ""),
    }


def _same_world(
    call: dict[str, Any],
    state: WorldState,
    speaker: str,
    condition: str,
    run: dict[str, Any],
) -> bool:
    if call["speaker"] != speaker:
        return False
    payload = _parse_json(_first_user(call))
    if not isinstance(payload, dict):
        return False
    body = payload.get("state")
    if not isinstance(body, dict):
        return False
    book = body.get("fiduciary")
    if not isinstance(book, dict):
        return False
    if body.get("date") != state.date.isoformat() and body.get("scenario_id") != state.scenario_id:
        return False
    if not _close(book.get("discretionary_remaining"), state.fiduciary.discretionary_remaining):
        return False
    expected_rent = (
        state.fiduciary.fixed_liabilities_upcoming[0].amount
        if state.fiduciary.fixed_liabilities_upcoming
        else None
    )
    liabilities = book.get("fixed_liabilities_upcoming")
    rent_known = expected_rent is not None and isinstance(liabilities, list) and liabilities
    if rent_known and not _close(liabilities[0].get("amount"), expected_rent):
        return False
    if speaker == "blame":
        if payload.get("committed_action_id") != run["chosen_action"]:
            return False
        showed_evidence = "evidence" in payload
        if showed_evidence != (condition == "D"):
            return False
    return True


def _final_call(calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not calls:
        return None
    return max(calls, key=lambda call: len(call["messages"]))


def _first_user(call: dict[str, Any]) -> str:
    for message in call["messages"]:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 0.02
    except (TypeError, ValueError):
        return False
