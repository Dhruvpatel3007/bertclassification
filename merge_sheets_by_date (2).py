#!/usr/bin/env python3
"""
Merge all date-named sheets of an Excel workbook into ONE sheet,
sorted in ascending order by the date taken from each sheet's name
(and then by the 'Sent' time inside each date).

Sheet names are expected like:  20-11-2026   (dd-mm-yyyy)

Usage:
    python merge_sheets_by_date.py input.xlsx
    python merge_sheets_by_date.py input.xlsx output.xlsx

Requires:  pip install pandas openpyxl
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# Date formats to try on the sheet names (add more if needed)
DATE_FORMATS = ["%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%y"]


def parse_sheet_date(sheet_name: str):
    """Convert a sheet name into a date. Returns pd.NaT if it is not a date."""
    name = sheet_name.strip()
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(name, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def to_time(value):
    """Normalise the 'Sent' value (time object / datetime / text) to datetime.time."""
    if pd.isna(value):
        return pd.NaT
    if hasattr(value, "time") and callable(value.time):   # datetime / Timestamp
        return value.time()
    if hasattr(value, "hour"):                            # datetime.time
        return value
    parsed = pd.to_datetime(str(value), errors="coerce")  # text like "16:47"
    return parsed.time() if not pd.isna(parsed) else pd.NaT


def merge_sheets(input_file: Path, output_file: Path) -> None:
    # sheet_name=None -> reads EVERY sheet into a dict {sheet_name: DataFrame}
    all_sheets = pd.read_excel(input_file, sheet_name=None, dtype=object)
    print(f"Found {len(all_sheets)} sheets in '{input_file.name}'")

    frames, skipped = [], []

    for sheet_name, df in all_sheets.items():
        sheet_date = parse_sheet_date(sheet_name)

        if pd.isna(sheet_date):
            skipped.append(sheet_name)
            print(f"  [SKIP] '{sheet_name}' - name is not a valid date")
            continue
        if df.empty:
            print(f"  [EMPTY] '{sheet_name}' - no rows")
            continue

        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")                      # drop fully blank rows
        df.insert(0, "Date", sheet_date)               # new Date column
        df["Source Sheet"] = sheet_name                # keep origin for traceability
        frames.append(df)
        print(f"  [OK]   '{sheet_name}' -> {len(df)} rows")

    if not frames:
        sys.exit("No valid date-named sheets found. Nothing to merge.")

    merged = pd.concat(frames, ignore_index=True)

    # Sort: Date ascending, then Sent time ascending (if the column exists)
    sort_cols = ["Date"]
    if "Sent" in merged.columns:
        merged["_sent_sort"] = merged["Sent"].apply(to_time)
        # NaT can't be compared with time objects -> use a sortable string
        merged["_sent_sort"] = merged["_sent_sort"].apply(
            lambda t: t.strftime("%H:%M:%S") if not pd.isna(t) else "99:99:99"
        )
        sort_cols.append("_sent_sort")

    merged = merged.sort_values(sort_cols, ascending=True, kind="stable")
    merged = merged.drop(columns=[c for c in ["_sent_sort"] if c in merged.columns])
    merged["Date"] = merged["Date"].dt.date            # show date only (no 00:00:00)
    merged = merged.reset_index(drop=True)

    # Write result
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="Merged", index=False)

    format_output(output_file)

    print(f"\nDone. {len(merged)} rows written to '{output_file}'")
    if skipped:
        print(f"Skipped non-date sheets: {skipped}")


def format_output(path: Path) -> None:
    """Basic formatting: bold header, freeze row, filter, column widths, date format."""
    wb = load_workbook(path)
    ws = wb["Merged"]

    header_fill = PatternFill("solid", start_color="1F4E78")
    for cell in ws[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Arial", size=10)
    for cell in ws["A"][1:]:                           # Date column
        cell.number_format = "DD-MM-YYYY"

    for col_idx, col in enumerate(ws.columns, start=1):
        longest = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(longest + 2, 60)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(path)


if __name__ == "__main__":
    # Loads variables from the .env file sitting next to this script
    load_dotenv(Path(__file__).with_name(".env"))

    # Priority: command-line argument > .env value
    input_value = sys.argv[1] if len(sys.argv) > 1 else os.getenv("INPUT_FILE")
    output_value = sys.argv[2] if len(sys.argv) > 2 else os.getenv("OUTPUT_FILE")

    if not input_value:
        sys.exit("INPUT_FILE is not set. Add it to the .env file "
                 "or pass it as: python merge_sheets_by_date.py input.xlsx")

    # .strip('"') so quotes pasted from Windows 'Copy as path' don't break it
    in_path = Path(input_value.strip().strip('"'))
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    out_path = (
        Path(output_value.strip().strip('"'))
        if output_value
        else in_path.with_name(f"{in_path.stem}_merged_sorted.xlsx")
    )
    merge_sheets(in_path, out_path)
