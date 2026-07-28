#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发测试脚本 - 读取 CSV 中的问题，对每一行两列分别并发调用 API 获取回答。

用法：
    python online_test/fuzzy_test.py
    python online_test/fuzzy_test.py --csv data/question_fuzzy.csv --max_workers 4
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_client import log, create_session, call_api, extract_complete_answer


@dataclass
class QueryResult:
    row: int
    question: str
    column: str
    session_id: str = ""
    answer: str = ""
    error: Optional[str] = None
    duration: float = 0.0


def ask_once(question: str, row: int, column: str) -> QueryResult:
    result = QueryResult(row=row, question=question, column=column)
    t0 = time.time()

    try:
        session_id = create_session()
        result.session_id = session_id

        resp = call_api(
            pod_name="anlynetmasterapp",
            url_path="",
            request_body={},
            session_id=session_id,
            content=question,
        )

        answer = extract_complete_answer(resp)
        result.answer = answer

    except Exception as e:
        result.error = str(e)
        log(f"[行{row}][{column}] 出错: {e}")

    finally:
        result.duration = time.time() - t0

    return result


def resolve_path(path: str, base_dir: str) -> str:
    """
    把相对路径转成基于项目根目录的绝对路径。
    当前脚本在 online_test/ 下，所以项目根目录是脚本目录的上一级。
    """
    if os.path.isabs(path):
        return os.path.normpath(path)

    return os.path.normpath(os.path.join(base_dir, "..", path))


def main():
    parser = argparse.ArgumentParser(description="并发测试：对 CSV 中两列问题分别调用 API")

    parser.add_argument(
        "--csv",
        default="data/question_fuzzy.csv",
        help="CSV 文件路径，默认 data/question_fuzzy.csv",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="最大并发数，默认 1",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="结果输出 CSV 路径；不指定则自动生成 fuzzy_results_时间戳.csv",
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = resolve_path(args.csv, script_dir)

    if not os.path.exists(csv_path):
        print(f"文件不存在: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_dict in reader:
            rows.append(row_dict)

    if not rows:
        print(f"CSV 为空，没有可执行的问题: {csv_path}")
        sys.exit(1)

    columns = list(rows[0].keys())

    target_columns = ["原始查询", "改写查询"]
    missing_columns = [col for col in target_columns if col not in columns]

    if missing_columns:
        print(f"CSV 缺少必要列: {missing_columns}")
        print(f"当前 CSV 列: {columns}")
        sys.exit(1)

    print(
        f"读取到 {len(rows)} 行问题，共最多 {len(rows) * len(target_columns)} 个请求"
        f"（目标列 {target_columns}）"
    )

    tasks = []
    for i, row_dict in enumerate(rows, start=1):
        for col in target_columns:
            question = row_dict.get(col, "").strip()
            if question:
                tasks.append((question, i, col))

    if not tasks:
        print("没有找到非空问题，退出。")
        sys.exit(1)

    if args.output:
        out_path = resolve_path(args.output, script_dir)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(csv_path),
            f"fuzzy_results_{ts}.csv",
        )

    max_workers = args.max_workers

    if max_workers <= 0:
        print(f"--max_workers 必须大于 0，当前值: {max_workers}")
        sys.exit(1)

    print(f"\n并发数: {max_workers}, 任务总数: {len(tasks)}")
    print(f"输出文件: {out_path}")

    results: list[QueryResult] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fut_map = {
            executor.submit(ask_once, question, row, col): (question, row, col)
            for question, row, col in tasks
        }

        for fut in as_completed(fut_map):
            question, row, col = fut_map[fut]

            try:
                res = fut.result()
            except Exception as e:
                res = QueryResult(
                    row=row,
                    question=question,
                    column=col,
                    error=str(e),
                )

            results.append(res)
            done_count += 1

            if res.error:
                print(f"  [{done_count}/{len(tasks)}] 行{row} [{col}] 失败: {res.error}")
            else:
                print(
                    f"  [{done_count}/{len(tasks)}] 行{row} [{col}] 完成 "
                    f"({res.duration:.1f}s)"
                )

    fieldnames = [
        "row",
        "column",
        "question",
        "answer",
        "session_id",
        "error",
        "duration",
    ]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(results, key=lambda x: (x.row, x.column)):
            writer.writerow(asdict(r))

    success_count = sum(1 for r in results if not r.error)

    print(f"\n结果已写入: {out_path}")
    print(f"成功: {success_count} / {len(results)}")

    print("\n结果摘要")
    for r in sorted(results, key=lambda x: (x.row, x.column)):
        status = "OK" if not r.error else "FAIL"

        if r.error:
            preview = r.error[:80]
        else:
            preview = r.answer[:80].replace("\n", " ")

        print(
            f"  [{status}] 行{r.row} {r.column}: "
            f"{r.question[:40]}... -> {preview}..."
        )


if __name__ == "__main__":
    main()
