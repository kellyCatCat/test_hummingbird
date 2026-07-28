#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话测试脚本 - 读取 CSV 中的多轮查询，同一个对话复用同一个 session 依次提问。

CSV 为长表格式，一行一轮，必须包含「对话ID」「轮次」「查询」三列：

    对话ID,轮次,查询
    1,1,查一下小赵庄东的告警
    1,2,那昨天呢
    2,1,最近一周哪个网元告警最多

同一「对话ID」下的所有轮次按「轮次」升序、**串行**发送，共用一个 session_id，
服务端因此能保留上下文；不同对话之间相互独立，可以并发。
所以 --max_workers 控制的是「并发对话数」，不是并发请求数。

某一轮失败时，该对话的后续轮次不再发送（上下文已断），会标记为跳过。

结果是**边跑边落盘**：每个对话一跑完就追加写入输出 CSV 并 fsync，中途 Ctrl+C
或进程被杀，已完成的对话都还在文件里。Ctrl+C 会取消尚未开始的对话，进行中的
对话在当前这一轮结束后停下，其已完成轮次同样会写入。全部跑完后文件会按
(对话ID, 轮次) 重排一次；中断的情况下文件保持完成顺序。

用法：
    python online_test/multi_turn_test.py
    python online_test/multi_turn_test.py --csv data/question_multi_turn.csv --max_workers 4
"""

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_client import (
    log,
    create_session,
    call_api,
    extract_complete_answer,
    extract_thinking,
    extract_tool_calls,
)


# Ctrl+C 后置位；各线程在每轮之间检查，尽快停下来
STOP = threading.Event()


@dataclass
class TurnResult:
    conv_id: str
    turn: int
    question: str
    session_id: str = ""
    answer: str = ""
    thinking: str = ""
    tool_calls: str = ""
    raw_response: str = ""
    error: Optional[str] = None
    duration: float = 0.0


def run_conversation(conv_id: str, turns: list) -> list:
    """
    跑完一个完整对话：建一次 session，然后按轮次串行提问。

    Args:
        conv_id: 对话 ID
        turns: [(轮次, 查询), ...]，已按轮次升序排好

    Returns:
        每一轮的 TurnResult 列表
    """
    results = []

    if STOP.is_set():
        return [
            TurnResult(
                conv_id=conv_id,
                turn=turn,
                question=question,
                error="跳过：收到中断信号",
            )
            for turn, question in turns
        ]

    try:
        session_id = create_session()
    except Exception as e:
        log(f"[对话{conv_id}] 创建会话失败: {e}")
        return [
            TurnResult(
                conv_id=conv_id,
                turn=turn,
                question=question,
                error=f"创建会话失败: {e}",
            )
            for turn, question in turns
        ]

    aborted = False

    for turn, question in turns:
        result = TurnResult(
            conv_id=conv_id,
            turn=turn,
            question=question,
            session_id=session_id,
        )

        if aborted:
            # 上一轮失败，上下文已断，后面几轮再发也没有意义
            result.error = "跳过：同一对话的前序轮次失败"
            results.append(result)
            continue

        if STOP.is_set():
            result.error = "跳过：收到中断信号"
            results.append(result)
            continue

        t0 = time.time()

        try:
            resp = call_api(
                pod_name="hummingbird",
                url_path="",
                request_body={},
                session_id=session_id,
                content=question,
            )

            result.answer = extract_complete_answer(resp)
            result.thinking = extract_thinking(resp)
            result.tool_calls = extract_tool_calls(resp)
            result.raw_response = resp.get("_raw", "")

        except Exception as e:
            result.error = str(e)
            aborted = True
            log(f"[对话{conv_id}][第{turn}轮] 出错: {e}")

        finally:
            result.duration = time.time() - t0

        results.append(result)

    return results


def resolve_path(path: str, base_dir: str) -> str:
    """
    把相对路径转成基于项目根目录的绝对路径。
    当前脚本在 online_test/ 下，所以项目根目录是脚本目录的上一级。
    """
    if os.path.isabs(path):
        return os.path.normpath(path)

    return os.path.normpath(os.path.join(base_dir, "..", path))


def load_conversations(rows: list) -> dict:
    """
    把长表的行按「对话ID」分组，并在组内按「轮次」升序排序。

    Returns:
        {对话ID: [(轮次, 查询), ...]}，对话顺序按其在 CSV 中首次出现的顺序
    """
    conversations = {}
    errors = []

    for line_no, row_dict in enumerate(rows, start=2):  # 表头占第 1 行
        conv_id = (row_dict.get("对话ID") or "").strip()
        turn_raw = (row_dict.get("轮次") or "").strip()
        question = (row_dict.get("查询") or "").strip()

        if not conv_id and not turn_raw and not question:
            continue  # 整行为空，跳过

        if not conv_id:
            errors.append(f"第 {line_no} 行: 对话ID 为空")
            continue

        if not question:
            errors.append(f"第 {line_no} 行: 查询 为空")
            continue

        try:
            turn = int(turn_raw)
        except ValueError:
            errors.append(f"第 {line_no} 行: 轮次 不是整数 -> {turn_raw!r}")
            continue

        conversations.setdefault(conv_id, []).append((turn, question))

    if errors:
        print("CSV 数据有问题：")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)

    for turns in conversations.values():
        turns.sort(key=lambda t: t[0])

    return conversations


def sort_key(conv_id: str, turn: int):
    """对话 ID 优先按数字排序，非数字的排在后面按字符串排。"""
    try:
        return (0, int(conv_id), "", turn)
    except ValueError:
        return (1, 0, conv_id, turn)


def main():
    parser = argparse.ArgumentParser(description="多轮对话测试：同一对话复用 session 串行提问")

    parser.add_argument(
        "--csv",
        default="data/question_multi_turn.csv",
        help="CSV 文件路径，默认 data/question_multi_turn.csv",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="最大并发【对话】数，默认 1；同一对话内部始终串行",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="结果输出 CSV 路径；不指定则自动生成 multi_turn_results_时间戳.csv",
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
        print(f"CSV 为空，没有可执行的对话: {csv_path}")
        sys.exit(1)

    columns = list(rows[0].keys())

    target_columns = ["对话ID", "轮次", "查询"]
    missing_columns = [col for col in target_columns if col not in columns]

    if missing_columns:
        print(f"CSV 缺少必要列: {missing_columns}")
        print(f"当前 CSV 列: {columns}")
        sys.exit(1)

    conversations = load_conversations(rows)

    if not conversations:
        print("没有找到有效对话，退出。")
        sys.exit(1)

    total_turns = sum(len(turns) for turns in conversations.values())

    print(f"读取到 {len(conversations)} 个对话，共 {total_turns} 轮查询")

    max_workers = args.max_workers

    if max_workers <= 0:
        print(f"--max_workers 必须大于 0，当前值: {max_workers}")
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
            f"multi_turn_results_{ts}.csv",
        )

    print(f"\n并发对话数: {max_workers}（同一对话内串行）")
    print(f"输出文件: {out_path}")

    # raw_response 整段报文很长，放最后一列，前面几列才好读
    fieldnames = [
        "conv_id",
        "turn",
        "question",
        "answer",
        "thinking",
        "tool_calls",
        "session_id",
        "error",
        "duration",
        "raw_response",
    ]

    results = []
    done_count = 0
    processed = set()
    interrupted = False

    with open(out_path, "w", encoding="utf-8-sig", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        out_file.flush()

        def handle_future(fut, conv_id, turns):
            """收下一个对话的结果，立刻追加落盘，并打印进度。"""
            nonlocal done_count

            if fut in processed:
                return

            processed.add(fut)

            try:
                conv_results = fut.result()
            except Exception as e:
                conv_results = [
                    TurnResult(
                        conv_id=conv_id,
                        turn=turn,
                        question=question,
                        error=str(e),
                    )
                    for turn, question in turns
                ]

            results.extend(conv_results)
            done_count += 1

            for r in conv_results:
                writer.writerow(asdict(r))

            # 每跑完一个对话就落盘，中途被打断也不会丢已完成的部分
            out_file.flush()
            os.fsync(out_file.fileno())

            ok = sum(1 for r in conv_results if not r.error)
            elapsed = sum(r.duration for r in conv_results)

            print(
                f"  [{done_count}/{len(conversations)}] 对话{conv_id} 完成 "
                f"{ok}/{len(conv_results)} 轮 ({elapsed:.1f}s) 已落盘"
            )

        executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            fut_map = {
                executor.submit(run_conversation, conv_id, turns): (conv_id, turns)
                for conv_id, turns in conversations.items()
            }

            try:
                for fut in as_completed(fut_map):
                    conv_id, turns = fut_map[fut]
                    handle_future(fut, conv_id, turns)

            except KeyboardInterrupt:
                interrupted = True
                STOP.set()

                cancelled = sum(1 for f in fut_map if f.cancel())

                print(
                    f"\n收到中断信号：已取消 {cancelled} 个尚未开始的对话，"
                    f"进行中的对话会在当前这一轮结束后停下……"
                )

                # 继续收尾：把进行中的对话已经跑出来的轮次也写进去
                for fut, (conv_id, turns) in fut_map.items():
                    if fut.cancelled():
                        continue
                    handle_future(fut, conv_id, turns)

        finally:
            executor.shutdown(wait=True)

    # 追加时是按完成顺序写的，最后按 (对话ID, 轮次) 重排一次。
    # 先写临时文件再原子替换，重排出问题也不会破坏已经落盘的数据。
    try:
        tmp_path = out_path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in sorted(results, key=lambda x: sort_key(x.conv_id, x.turn)):
                writer.writerow(asdict(r))

        os.replace(tmp_path, out_path)

    except Exception as e:
        print(f"结果重排失败，文件保持完成顺序（数据未丢失）: {e}")

    success_count = sum(1 for r in results if not r.error)

    if interrupted:
        print(f"\n已中断，部分结果已写入: {out_path}")
    else:
        print(f"\n结果已写入: {out_path}")

    print(f"成功: {success_count} / {len(results)} 轮")

    print("\n结果摘要")
    for r in sorted(results, key=lambda x: sort_key(x.conv_id, x.turn)):
        status = "OK" if not r.error else "FAIL"

        if r.error:
            preview = r.error[:80]
        else:
            preview = r.answer[:80].replace("\n", " ")

        print(
            f"  [{status}] 对话{r.conv_id} 第{r.turn}轮: "
            f"{r.question[:40]}... -> {preview}..."
        )


if __name__ == "__main__":
    main()
