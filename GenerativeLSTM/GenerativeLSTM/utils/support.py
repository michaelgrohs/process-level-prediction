"""Minimal reimplementation of utils.support for GenerativeLSTM."""
import csv
import json
import os
from datetime import datetime


def folder_id() -> str:
    return datetime.now().strftime("f%Y%m%d%H%M%S%f")


def file_id(prefix: str = "", extension: str = ".csv") -> str:
    return f"{prefix}{datetime.now().strftime('%Y%m%d%H%M%S%f')}{extension}"


def create_json(data: dict, path: str) -> None:
    def _default(obj):
        try:
            return int(obj)
        except (TypeError, ValueError):
            return str(obj)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=_default)


def create_csv_file(data: list, path: str, mode: str = "w") -> None:
    if not data:
        return
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writerows(data)


def create_csv_file_header(data: list, path: str) -> None:
    if not data:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def create_file_from_list(data: list, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def reduce_list(string: str, dtype: str = "str") -> list:
    if not string or string.strip() in ("", "[]", "None"):
        return []
    string = string.strip().strip("[]")
    items = [s.strip().strip("'\"") for s in string.split(",") if s.strip()]
    if dtype == "int":
        return [int(x) for x in items]
    if dtype == "float":
        return [float(x) for x in items]
    return items


def print_progress(pct: float, message: str = "") -> None:
    print(f"  {message} {pct:.0f}%")


def print_done_task() -> None:
    print("  Done.")


def print_performed_task(message: str = "") -> None:
    print(f"  {message}")
