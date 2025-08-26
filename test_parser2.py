# test_parser.py
import os
import importlib.util
import traceback
import pandas as pd
import pytest

# You can override target by setting env var TARGET (e.g. TARGET=sbi pytest)
TARGET = os.getenv("TARGET", "icici").lower()
PARSER_PATH = f"custom_parsers/{TARGET}_parser.py"
PDF_PATH = f"data/{TARGET}/{TARGET}_sample.pdf"
CSV_PATH = f"data/{TARGET}/{TARGET}_sample.csv"

def import_parser_module(path: str):
    """Dynamically import the parser module from a filepath."""
    if not os.path.exists(path):
        pytest.skip(f"Parser file not found at {path}")
    spec = importlib.util.spec_from_file_location("generated_parser", path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise RuntimeError("Could not load module spec.loader")
    loader.exec_module(module)
    return module

def safe_parse(module, pdf_path: str):
    """Call module.parse(pdf_path) and return the DataFrame (or raise helpful assertion)."""
    if not hasattr(module, "parse"):
        pytest.fail("Parser module does not define a parse(pdf_path) function")
    parse_fn = getattr(module, "parse")
    try:
        df = parse_fn(pdf_path)
    except Exception as e:
        tb = traceback.format_exc()
        pytest.fail(f"Calling parse(pdf_path) raised an exception: {e}\n\n{tb}")
    return df

def test_parser_file_exists():
    assert os.path.exists(PARSER_PATH), f"Parser file not found: {PARSER_PATH}"

def test_parse_returns_nonempty_dataframe():
    module = import_parser_module(PARSER_PATH)
    df = safe_parse(module, PDF_PATH)
    assert isinstance(df, pd.DataFrame), "parse did not return a pandas.DataFrame"
    assert len(df) > 0, "DataFrame is empty - parser failed to extract transactions"

def test_columns_match_csv_schema():
    assert os.path.exists(CSV_PATH), f"CSV sample not found: {CSV_PATH}"
    sample = pd.read_csv(CSV_PATH)
    expected_cols = list(sample.columns)
    module = import_parser_module(PARSER_PATH)
    df = safe_parse(module, PDF_PATH)
    got_cols = list(df.columns)
    assert got_cols == expected_cols, (
        f"Column order/names mismatch.\nExpected: {expected_cols}\nGot:      {got_cols}\n"
        "Make sure your parser returns EXACT columns in the same order as the CSV."
    )

def test_debit_credit_preserve_empty_strings():
    module = import_parser_module(PARSER_PATH)
    df = safe_parse(module, PDF_PATH)
    # find debit/credit columns (case-insensitive)
    debit_col = next((c for c in df.columns if 'debit' in c.lower()), None)
    credit_col = next((c for c in df.columns if 'credit' in c.lower()), None)
    assert debit_col is not None, "Debit column not found (column name must include 'debit')"
    assert credit_col is not None, "Credit column not found (column name must include 'credit')"
    # There should be at least one empty string in each (not NaN/None)
    empty_debits = (df[debit_col] == '').sum()
    empty_credits = (df[credit_col] == '').sum()
    assert empty_debits > 0, "No empty strings found in Debit column — ensure missing amounts are '' (empty string)"
    assert empty_credits > 0, "No empty strings found in Credit column — ensure missing amounts are '' (empty string)"

def test_numeric_like_columns_convertible():
    module = import_parser_module(PARSER_PATH)
    df = safe_parse(module, PDF_PATH)
    # detect numeric-like columns by name
    numeric_cols = [c for c in df.columns if any(w in c.lower() for w in ['amt', 'amount', 'debit', 'credit', 'balance'])]
    for c in numeric_cols:
        non_empty = df[c][df[c] != '']
        if len(non_empty) == 0:
            # nothing to check for this column
            continue
        # this will raise if conversion fails
        try:
            pd.to_numeric(non_empty, errors='raise')
        except Exception as e:
            pytest.fail(f"Column '{c}' contains non-numeric values in non-empty rows: {e}")

def test_row_count_reasonable_vs_csv():
    assert os.path.exists(CSV_PATH), f"CSV sample not found: {CSV_PATH}"
    sample = pd.read_csv(CSV_PATH)
    expected_rows = len(sample)
    module = import_parser_module(PARSER_PATH)
    df = safe_parse(module, PDF_PATH)
    assert len(df) >= max(1, int(expected_rows * 0.7)), (
        f"Too few rows extracted: {len(df)} vs expected ~{expected_rows} "
        "(require >= 70% of expected rows). Check multi-page handling and regex patterns."
    )
