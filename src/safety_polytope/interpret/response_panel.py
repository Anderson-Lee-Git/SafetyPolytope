import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


DEFAULT_JSON_PATH = (
    "/scratch/gpfs/KOROLOVA/cl6486/SafetyPolytope/outputs/beaver_tails/2025-10-20/"
    "eval_generation-03-48-55/output_data.json"
)


@st.cache_data(show_spinner=False)
def load_output_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def format_safety_label(is_safe: Optional[bool]) -> str:
    if is_safe is None:
        return ":gray[UNKNOWN]"
    return ":green[SAFE]" if bool(is_safe) else ":red[UNSAFE]"


def get_min_length(arrays: List[List[Any]]) -> int:
    lengths = [len(a) for a in arrays if a is not None]
    return min(lengths) if lengths else 0


def main() -> None:
    st.set_page_config(page_title="Response Panel", layout="wide")
    st.title("Response Comparison Panel")
    st.caption(
        "Visualize side-by-side original vs calibrated responses with safety labels."
    )

    with st.sidebar:
        st.header("Data Source")
        default_path = DEFAULT_JSON_PATH
        file_path = st.text_input(
            "Path to output_data.json",
            value=default_path,
            help="Absolute path to the evaluation JSON file.",
        )

    if not file_path:
        st.info("Provide a JSON file path in the sidebar to begin.")
        return

    path_obj = Path(file_path)
    if not path_obj.exists():
        st.error(f"File not found: {file_path}")
        return

    try:
        data = load_output_json(str(path_obj))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load JSON: {exc}")
        return

    # Extract fields expected from the data, mirroring main.py
    original_responses: Optional[List[str]] = data.get("original_responses")
    safe_responses: Optional[List[str]] = data.get("safe_responses")
    original_safe_labels: Optional[List[bool]] = data.get("original_safe_labels")
    safe_labels: Optional[List[bool]] = data.get("safe_labels")
    labels: Optional[List[Any]] = data.get("labels")
    categories: Optional[List[Any]] = data.get("categories")

    if any(
        v is None
        for v in [original_responses, safe_responses, original_safe_labels, safe_labels]
    ):
        st.error(
            "JSON is missing required keys. Expected keys: "
            "'original_responses', 'safe_responses', 'original_safe_labels', 'safe_labels'"
        )
        return

    # Type hints for mypy/linters
    assert isinstance(original_responses, list)
    assert isinstance(safe_responses, list)
    assert isinstance(original_safe_labels, list)
    assert isinstance(safe_labels, list)

    # Determine consistent length across arrays present
    arrays_for_length = [
        original_responses,
        safe_responses,
        original_safe_labels,
        safe_labels,
    ]
    if labels is not None:
        arrays_for_length.append(labels)
    if categories is not None:
        arrays_for_length.append(categories)
    num_items = get_min_length(arrays_for_length)

    if num_items == 0:
        st.warning("No items found to display.")
        return

    # Summary
    with st.expander("Dataset summary", expanded=True):
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Original responses", len(original_responses))
        col_b.metric("Calibrated responses", len(safe_responses))
        col_c.metric("Original safety labels", len(original_safe_labels))
        col_d.metric("Calibrated safety labels", len(safe_labels))
        if labels is not None:
            col_e.metric("Labels", len(labels))
        else:
            col_e.metric("Labels", 0)

        if len(set(len(a) for a in arrays_for_length)) != 1:
            st.info(
                "Lengths differ across arrays; using the minimum common length for indexing."
            )

    # Safety filters
    with st.sidebar:
        st.header("Filters")
        orig_choice = st.selectbox(
            "Original safety",
            ["Any", "Safe", "Unsafe"],
            index=0,
        )
        cal_choice = st.selectbox(
            "Calibrated safety",
            ["Any", "Safe", "Unsafe"],
            index=0,
        )

    def choice_to_bool(choice: str) -> Optional[bool]:
        if choice == "Any":
            return None
        if choice == "Safe":
            return True
        return False

    orig_filter = choice_to_bool(orig_choice)
    cal_filter = choice_to_bool(cal_choice)

    filtered_indices = [
        i
        for i in range(num_items)
        if (orig_filter is None or bool(original_safe_labels[i]) == orig_filter)
        and (cal_filter is None or bool(safe_labels[i]) == cal_filter)
    ]

    if len(filtered_indices) == 0:
        st.warning("No samples match the selected safety filters.")
        return

    # Index selection within filtered subset
    with st.sidebar:
        st.header("Viewer")
        pos = st.number_input(
            "Position (within filtered)",
            min_value=0,
            max_value=len(filtered_indices) - 1,
            value=0,
            step=1,
        )
        st.caption(f"{len(filtered_indices)} of {num_items} samples match filters")

    index = filtered_indices[int(pos)]

    # Details for selected index
    st.subheader(f"Item {index}")
    meta_cols = st.columns(3)
    meta_cols[0].markdown(
        f"**Original safety:** {format_safety_label(original_safe_labels[index])}"
    )
    meta_cols[1].markdown(
        f"**Calibrated safety:** {format_safety_label(safe_labels[index])}"
    )
    if labels is not None:
        meta_cols[2].markdown(f"**Label:** `{labels[index]}`")
    elif categories is not None:
        meta_cols[2].markdown(f"**Category:** `{categories[index]}`")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original")
        st.markdown(
            (
                f"{original_responses[index]}"
                if original_responses[index] is not None
                else ""
            ),
        )

    with col2:
        st.markdown("### Calibrated")
        st.markdown(
            f"{safe_responses[index]}" if safe_responses[index] is not None else ""
        )


if __name__ == "__main__":
    main()
