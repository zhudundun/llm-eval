"""
Deterministic pre-checks for the LLM judge (Step 3, §8 of the design doc).

These checks are cheap, deterministic, and ground the LLM judge with concrete
facts about the run: did the code execute, did it use the expected CSV, did it
reference columns that do not exist, does the final answer look hardcoded, and
do the reported answer values actually appear in stdout.

All functions are pure and return JSON-serializable values (plain bool / list /
str), so the output can be embedded directly in a judge prompt or persisted.
"""
from __future__ import annotations

import re

from src.benchmark.task import AgentResponse, DABenchTask

# ---------------------------------------------------------------------------
# Column-reference detection
# ---------------------------------------------------------------------------

# Bracket indexing with a string key:  df['col'] / df["col"] / x[ "col" ]
# Captures the quoted contents of any [...] subscript.  This intentionally also
# catches non-df subscripts (e.g. a dict literal key); that's acceptable for a
# best-effort heuristic and errs toward over-reporting referenced names, which
# the unknown-column filter then narrows back down against the real schema.
_BRACKET_KEY_RE = re.compile(r"""\[\s*(['"])(?P<col>.*?)\1\s*\]""")

# Attribute access on a `df`-like name:  df.Close  /  data.col_name  /  df1.x
# Requires the base to look like a dataframe handle so we don't pick up things
# like ``pd.read_csv`` or ``np.mean``.  The attribute must be a valid identifier
# and is excluded if it's a common pandas method/attribute.
_DF_ATTR_RE = re.compile(
    r"""\b(?P<base>[A-Za-z_]\w*?(?:df|data|frame)\w*|df\w*|data\w*)"""
    r"""\.(?P<col>[A-Za-z_]\w*)""",
    re.IGNORECASE,
)

# Pandas attributes/methods that are NOT column names — exclude from df.attr matches.
_PANDAS_NON_COLUMNS = frozenset(
    {
        "head", "tail", "describe", "info", "columns", "index", "values",
        "shape", "dtypes", "dtype", "loc", "iloc", "at", "iat", "T",
        "mean", "median", "mode", "sum", "min", "max", "std", "var", "count",
        "nunique", "unique", "value_counts", "groupby", "agg", "apply", "map",
        "merge", "join", "concat", "sort_values", "sort_index", "drop",
        "dropna", "fillna", "isna", "isnull", "notna", "notnull", "astype",
        "copy", "reset_index", "set_index", "rename", "corr", "cov", "round",
        "abs", "to_csv", "to_dict", "to_numpy", "to_list", "tolist", "size",
        "empty", "ndim", "axes", "name", "str", "dt", "cat", "plot", "hist",
        "pivot", "pivot_table", "melt", "stack", "unstack", "nlargest",
        "nsmallest", "quantile", "cumsum", "cumprod", "diff", "pct_change",
        "rolling", "expanding", "shift", "rank", "duplicated", "drop_duplicates",
        "sample", "query", "filter", "where", "mask", "replace", "clip",
        "idxmax", "idxmin", "any", "all", "sem", "skew", "kurt", "prod",
        "read_csv", "DataFrame", "Series", "iterrows", "itertuples", "items",
    }
)


def detect_referenced_columns(code: str) -> set[str]:
    """Extract candidate column names referenced in ``code``.

    Matches the common pandas access patterns, best-effort via regex:

    * ``df['x']`` / ``df["x"]`` (bracket indexing with a string key)
    * ``["x"]`` (any string-key subscript)
    * ``df.x`` / ``data.x`` (attribute access on a dataframe-like handle)

    Returns a set of the extracted names.  Names that are clearly pandas
    methods/attributes (e.g. ``df.mean``) are excluded; common DataFrame
    methods are not treated as columns.
    """
    if not code:
        return set()

    referenced: set[str] = set()

    # Bracket-key subscripts: df['col'], ["col"], etc.
    for match in _BRACKET_KEY_RE.finditer(code):
        col = match.group("col")
        if col:
            referenced.add(col)

    # Attribute access on a df-like base: df.Close, data.price
    for match in _DF_ATTR_RE.finditer(code):
        col = match.group("col")
        if col and col not in _PANDAS_NON_COLUMNS:
            referenced.add(col)

    return referenced


def detect_unknown_columns(code: str, dataset_summary: dict) -> list[str]:
    """Return referenced columns that are not present in the dataset schema.

    If ``dataset_summary`` has no ``"columns"`` key (e.g. it's an error dict or
    the CSV was unavailable), returns ``[]`` since we cannot validate anything.
    """
    columns = dataset_summary.get("columns") if dataset_summary else None
    if not columns:
        return []

    known = {
        col.get("name")
        for col in columns
        if isinstance(col, dict) and col.get("name") is not None
    }

    referenced = detect_referenced_columns(code)
    unknown = sorted(name for name in referenced if name not in known)
    return unknown


def detect_hardcoded_answer(code: str, answer: dict) -> bool:
    """Heuristic: True if any answer VALUE appears verbatim as a literal in code.

    Used to flag potential hardcoding (the agent writing the answer directly
    rather than computing it).  Trivially short values (len < 2 after casting to
    str) are ignored to avoid noise from single digits/characters.
    """
    if not code or not answer:
        return False

    for value in answer.values():
        if value is None:
            continue
        value_str = str(value).strip()
        if len(value_str) < 2:
            continue
        if value_str in code:
            return True

    return False


def _answer_values_in_stdout(answer: dict, stdout: str) -> bool:
    """True if every (non-empty) answer value appears as a substring of stdout.

    An empty answer dict trivially returns True (there is nothing to find).
    """
    if not answer:
        return True

    haystack = stdout or ""
    for value in answer.values():
        if value is None:
            value_str = ""
        else:
            value_str = str(value).strip()
        if not value_str:
            # An empty/None value can't be located; treat as not present.
            return False
        if value_str not in haystack:
            return False

    return True


def run_prechecks(
    task: DABenchTask,
    response: AgentResponse,
    dataset_summary: dict,
) -> dict:
    """Cheap deterministic checks that ground the LLM judge.

    Returns a dict with keys:

    * ``execution_success`` (bool): ``response.execution_result == "success"``
    * ``stderr_present`` (bool): stderr is non-empty after stripping
    * ``final_answer_empty`` (bool): ``not response.answer``
    * ``code_present`` (bool): code is non-empty after stripping
    * ``timeout`` (bool): execution timed out, or ``TimeoutError`` in stderr
    * ``used_expected_csv`` (bool): ``task.file_name`` appears in the code
    * ``referenced_unknown_columns`` (list[str]): referenced columns not in schema
    * ``possible_hardcoded_answer`` (bool): answer values appear as code literals
    * ``answer_value_in_stdout`` (bool): every answer value appears in stdout
    """
    code = response.code or ""
    stderr = response.stderr or ""
    stdout = response.stdout or ""
    answer = response.answer or {}

    timeout = (
        response.execution_result == "timeout"
        or "TimeoutError" in stderr
    )

    used_expected_csv = bool(task.file_name) and (task.file_name in code)

    return {
        "execution_success": response.execution_result == "success",
        "stderr_present": bool(stderr.strip()),
        "final_answer_empty": not bool(answer),
        "code_present": bool(code.strip()),
        "timeout": bool(timeout),
        "used_expected_csv": bool(used_expected_csv),
        "referenced_unknown_columns": detect_unknown_columns(code, dataset_summary),
        "possible_hardcoded_answer": detect_hardcoded_answer(code, answer),
        "answer_value_in_stdout": _answer_values_in_stdout(answer, stdout),
    }
