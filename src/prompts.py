"""Class names, display names, and Prompt Engineering (PE) prompt lists.

For each dataset we keep two prompt lists used to build the CLIP text classifier:

* ``baseline`` -- the DriveCLIP-style naive prompts (PE disabled).
* ``pe`` -- the Prompt Engineering prompts proposed in the paper (PE enabled).

``display_names`` are short human-readable class names used only for reporting
(CSV columns, confusion-matrix labels) and do not affect the CLIP text encoder.

The ordering of every list follows the dataset class index 0..9.
"""

from __future__ import annotations

# A single template is used, following the reference implementation.
TEMPLATES = ["an image of a person {}."]


# --------------------------------------------------------------------------- #
# SAM-DD
# --------------------------------------------------------------------------- #
SAM_DD_DISPLAY_NAMES = [
    "driving safely",
    "drinking water",
    "talking - left",
    "talking - right",
    "texting - left",
    "texting - right",
    "touching hairs",
    "adjusting glasses",
    "reaching behind",
    "dropping the head",
]

# DriveCLIP-style baseline prompts (PE off).
SAM_DD_BASELINE_PROMPTS = [
    "driving safely",
    "drinking water while driving",
    "talking to the phone on left hand while driving",
    "talking to the phone on right hand while driving",
    "texting on the phone with left hand while driving",
    "texting on the phone with right hand while driving",
    "touching hairs with hand while driving",
    "adjusting glasses with hand while driving",
    "reaching behind while driving",
    "dropping the head while driving",
]

# Prompt Engineering prompts (PE on). Changes w.r.t. the baseline:
#   - "driving safely"            -> "holding steering wheel with both hands while driving"
#   - drop the right-hand laterality on phone classes 3 and 5
#   - "touching hairs"            -> "touching head"
#   - "adjusting glasses"         -> "touching glasses"
#   - "reaching behind"           -> "looking at us"
#   - "dropping the head"         -> "keeping the head down"
SAM_DD_PE_PROMPTS = [
    "holding steering wheel with both hands while driving",
    "drinking water while driving",
    "talking to the phone on left hand while driving",
    "talking to the phone on hand while driving",
    "texting on the phone with left hand while driving",
    "texting on the phone with hand while driving",
    "touching head with hand while driving",
    "touching glasses with hand while driving",
    "looking at us while driving",
    "keeping the head down",
]


# --------------------------------------------------------------------------- #
# StateFarm
# --------------------------------------------------------------------------- #
STATEFARM_DISPLAY_NAMES = [
    "safe driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]

# DriveCLIP-style baseline prompts (PE off).
STATEFARM_BASELINE_PROMPTS = [
    "safely driving",
    "texting with the right hand while driving",
    "talking on the phone with the right hand while driving",
    "texting with the left hand while driving",
    "talking on the phone with the left hand while driving",
    "operating the car radio while driving",
    "drinking while driving",
    "reaching behind to the back seat while driving",
    "doing hair and makeup while driving",
    "talking to a passenger while driving",
]

# Prompt Engineering prompts (PE on).
STATEFARM_PE_PROMPTS = [
    "holding steering wheel with both hands while driving",
    "texting with the hand while driving",
    "talking on the phone with the hand while driving",
    "texting with the left hand while driving",
    "talking on the phone with the left hand while driving",
    "touching the dashboard while driving",
    "drinking water while driving",
    "reaching behind to the back seat while driving",
    "touching head and face while driving",
    "talking to a passenger while driving",
]


_REGISTRY = {
    "sam-dd": {
        "display_names": SAM_DD_DISPLAY_NAMES,
        "baseline": SAM_DD_BASELINE_PROMPTS,
        "pe": SAM_DD_PE_PROMPTS,
    },
    "statefarm": {
        "display_names": STATEFARM_DISPLAY_NAMES,
        "baseline": STATEFARM_BASELINE_PROMPTS,
        "pe": STATEFARM_PE_PROMPTS,
    },
}


def get_display_names(dataset: str) -> list[str]:
    """Return the human-readable class names for ``dataset``."""
    return list(_REGISTRY[_norm(dataset)]["display_names"])


def get_prompts(dataset: str, enable_pe: bool) -> list[str]:
    """Return the CLIP prompt list for ``dataset``.

    When ``enable_pe`` is True the Prompt Engineering list is returned,
    otherwise the DriveCLIP-style baseline list.
    """
    key = "pe" if enable_pe else "baseline"
    return list(_REGISTRY[_norm(dataset)][key])


def get_templates() -> list[str]:
    """Return the prompt template list."""
    return list(TEMPLATES)


def _norm(dataset: str) -> str:
    d = dataset.lower().replace("_", "-")
    if d in ("statefarm", "state-farm"):
        return "statefarm"
    if d in ("sam-dd", "samdd"):
        return "sam-dd"
    raise KeyError(f"Unknown dataset '{dataset}'. Expected 'sam-dd' or 'statefarm'.")
