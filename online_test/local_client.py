#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本：V1

远程容器化环境 API 调用工具
通过 SSH 连接到远程服务器，执行 kubectl 命令在 Pod 中调用 蜂鸟

使用说明
1.调用接口需要指定 pod 名称，通常业务 api 和 sql 接口 pod 为 hummingbird
2.SSH 连接信息通过环境变量配置，见 .env.example：
    export SSH_HOST=<右侧环境 ip>
    export SSH_PASSWORD=<sopuser 密码>
    export SSH_SU_PASSWORD=<root 密码>
3.DEBUG 字段用于控制日志打印，可以关掉（export SSH_DEBUG=0）
4.日志保存在 online_test/logs/ 目录下
5.流式消息 API：传入 session_id 和 content 参数
6.普通 API：传入 url_path 和 method 参数
"""

import json
import base64
import os
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

import paramiko


@dataclass
class SSHConfig:
    """SSH 配置对象"""
    host: str
    port: int
    username: str
    password: str
    su_password: str


# SSH 配置 - 先用 sopuser 登录，再 su 切换到 root
# 敏感信息一律从环境变量读取，不要写死在代码里（本仓库是公开仓库）
SSH_HOST = os.getenv("SSH_HOST", "")  # 右侧环境 ip，联系提交人
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_USERNAME = os.getenv("SSH_USERNAME", "sopuser")
SSH_PASSWORD = os.getenv("SSH_PASSWORD", "")
SSH_SU_PASSWORD = os.getenv("SSH_SU_PASSWORD", "")  # root 密码，用于 su 切换
DEBUG = os.getenv("SSH_DEBUG", "1") not in ("0", "false", "False")  # 调试模式开关

# 日志配置
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"local_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def log(msg: str):
    """打印调试日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [DEBUG] {msg}"
    print(log_line, flush=True)
    if DEBUG:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')


def exec_as_root(client, command: str, su_password: str) -> tuple:
    """使用 su 切换到 root 执行命令（通过 base64 避免引号嵌套问题）"""
    log(f"执行命令: {command}")

    # 使用 base64 编码命令，避免引号嵌套问题
    cmd_base64 = base64.b64encode(command.encode('utf-8')).decode('utf-8')

    # 构建命令：先 echo 密码，再用 su 执行，最后用 base64 -d 解码执行
    su_cmd = f'echo "{su_password}" | su - root -c "echo {cmd_base64} | base64 -d | bash"'
    log(f"完整 su 命令 (base64): {su_cmd[:200]}...")

    stdin, stdout, stderr = client.exec_command(su_cmd, timeout=120)

    stdout_data = stdout.read().decode("utf-8")
    stderr_data = stderr.read().decode("utf-8")

    log(f"stdout: {stdout_data[:500]}")
    log(f"stderr: {stderr_data[:500]}")

    return stdout_data.strip(), stderr_data.strip()


def _get_full_pod_name(
        client,
        pod_name: str,
        namespace: str,
        ssh_su_password: str,
) -> str:
    """获取完整的 Pod 名称"""
    log(f"步骤1: 获取 Pod 列表, namespace={namespace}, pod_name={pod_name}")
    get_pod_cmd = f"kubectl get pod -n {namespace} -o wide | grep {pod_name}"
    pod_output, error = exec_as_root(client, get_pod_cmd, ssh_su_password)

    if error and not pod_output:
        raise ValueError(f"获取 Pod 列表失败: {error}")

    if not pod_output:
        raise ValueError(f"未找到 Pod: {pod_name} in namespace {namespace}")

    # 解析 pod 名称（取第一列）
    full_pod_name = pod_output.split()[0]
    log(f"找到 Pod: {full_pod_name}")
    return full_pod_name


def build_curl_command(
        url_path: str,
        request_body: Dict[str, Any],
        full_pod_name: str,
        namespace: str,
) -> str:
    """构建 kubectl exec curl 命令"""
    # POST 请求：将 request_body 转换为单行 JSON
    body_str = json.dumps(request_body, ensure_ascii=False)
    # 关键：参考命令中使用 ' 转义单引号，我们也这样做
    body_str = body_str.replace("'", "\\u0027")

    log(f"请求体: {body_str}")

    # 构建内层的 curl 命令
    inner_curl = (
        f"curl -X POST "
        f"-H 'User-Agent:EnvManager' "
        f"-H 'Accept:*/*' "
        f"-H 'Content-Type:application/json' "
        f"-H 'X-User-Id:1' "
        f"--unix-socket /opt/sidecar/ir/http.sock "
        f"-d '{body_str}' "
        f"'http://localhost{url_path}'"
    )

    # 用 base64 编码 curl 命令，避免 shell 特殊字符解析问题
    inner_curl_base64 = base64.b64encode(inner_curl.encode('utf-8')).decode('utf-8')

    # 构建最终的 kubectl exec 命令
    curl_cmd = (
        f'kubectl exec -i -n {namespace} {full_pod_name} -- bash -c '
        f'"echo {inner_curl_base64} | base64 -d | bash"'
    )
    return curl_cmd


def parse_sse_response(response: str) -> list:
    """
    解析 SSE 流式响应

    响应格式示例：
        event: message_sent
        data: {"sessionId":"sess_123"}

        event: stream_event
        data: {"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}}

        event: result
        data: {"type":"result","subtype":"success","session_id":"sess_123"}

    Returns:
        解析后的事件列表，每个事件包含 event 和 data 字段
    """
    events = []
    lines = response.split('\n')
    current_event = None
    current_data = []

    for line in lines:
        if line.startswith('event:'):
            current_event = line[6:].strip()
        elif line.startswith('data:'):
            current_data.append(line[5:].strip())
        elif line == '':
            # 空行表示事件结束
            if current_event is not None:
                data_str = ''.join(current_data)
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = data_str
                events.append({
                    'event': current_event,
                    'data': data
                })
                current_event = None
                current_data = []

    return events


def parse_response(response: str) -> Dict[str, Any]:
    """解析 API 响应"""
    # 检查是否是 SSE 格式（包含 "event:" 和 "data:" 前缀）
    if 'event:' in response and 'data:' in response:
        events = parse_sse_response(response)
        return {'events': events}

    # 尝试解析为 JSON
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        return {'raw_response': response}
    return result


def call_api(
        url_path: str,
        request_body: Dict[str, Any],
        pod_name: str = "hummingbird",
        method: str = None,
        namespace: str = "nce",
        ssh_config: SSHConfig = None,
        session_id: str = None,
        content: str = None,
) -> Dict[str, Any]:
    """
    调用远程 Pod 的 API 并返回结果

    Args:
        pod_name: Pod 名称（部分匹配即可）
        url_path: API 路径
        request_body: 请求 JSON 体
        method: 请求方法 GET/POST，若为 None 则自动从 api_info.json 判断
        namespace: kubernetes namespace
        ssh_config: SSH 配置对象
        session_id: 会话 ID（用于 /api/sessions/:sessionId/messages/stream）
        content: 消息内容（用于 /api/sessions/:sessionId/messages/stream）

    Returns:
        解析后的响应 JSON
    """
    if ssh_config is None:
        missing = [
            name
            for name, value in (
                ("SSH_HOST", SSH_HOST),
                ("SSH_PASSWORD", SSH_PASSWORD),
                ("SSH_SU_PASSWORD", SSH_SU_PASSWORD),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"缺少环境变量: {', '.join(missing)}；"
                f"请参考 .env.example 设置后再运行，或显式传入 ssh_config"
            )

        ssh_config = SSHConfig(
            host=SSH_HOST,
            port=SSH_PORT,
            username=SSH_USERNAME,
            password=SSH_PASSWORD,
            su_password=SSH_SU_PASSWORD,
        )

    # 如果提供了 session_id 和 content，使用流式消息 API
    if session_id is not None and content is not None:
        url_path = f"/api/sessions/{session_id}/messages/stream"
        request_body = {"content": content}
        method = "POST"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        log(f"连接 SSH: {ssh_config.host}:{ssh_config.port} 用户: {ssh_config.username}")
        client.connect(
            hostname=ssh_config.host,
            port=ssh_config.port,
            username=ssh_config.username,
            password=ssh_config.password,
            timeout=30,
        )
        log("SSH 连接成功")

        full_pod_name = _get_full_pod_name(client, pod_name, namespace, ssh_config.su_password)

        log(f"步骤2: 调用 API, method={method}, url_path={url_path}")

        curl_cmd = build_curl_command(url_path, request_body, full_pod_name, namespace)

        log(f"curl 命令: {curl_cmd[:200]}...")

        log("执行 curl 命令...")
        response, error = exec_as_root(client, curl_cmd, ssh_config.su_password)

        if error and not response:
            raise RuntimeError(f"API 调用失败: {error}")

        log(f"API 响应: {response}")

        result = parse_response(response)

    finally:
        client.close()
        log("SSH 连接已关闭")

    return result


def create_session(pod_name: str = "hummingbird", namespace: str = "nce") -> str:
    """
    创建会话并返回 session_id

    Args:
        pod_name: Pod 名称
        namespace: kubernetes namespace

    Returns:
        session_id 字符串
    """
    log("========== 创建会话 ==========")
    result = call_api(
        pod_name=pod_name,
        url_path="/api/sessions",
        request_body={},
        method="POST",
        namespace=namespace,
    )

    if 'events' in result:
        # 解析 SSE 响应
        for event in result['events']:
            if event['event'] == 'message_sent' and 'sessionId' in event['data']:
                session_id = event['data']['sessionId']
                log(f"创建会话成功: {session_id}")
                return session_id

    # 兼容非 SSE 响应
    session_id = result.get('sessionId') or result.get('raw_response', '')
    log(f"创建会话成功: {session_id}")
    return session_id


def extract_complete_answer(result: dict) -> str:
    """
    从 API 返回的拼装数据中，直接提取并拼接出完整的回答字符串
    """
    full_answer = []

    # 1. 获取事件列表
    events = result.get('events', [])

    for item in events:
        # 2. 只关心类型为 stream_event 的事件
        if item.get('event') == 'stream_event':
            data = item.get('data', {})
            inner_event = data.get('event', {})

            # 3. 过滤出真正的文本增量片段 (content_block_delta)
            if inner_event.get('type') == 'content_block_delta':
                # 4. 安全地层层获取 text
                text = inner_event.get('delta', {}).get('text', '')
                full_answer.append(text)

    # 5. 将所有碎片拼接成一句话返回
    return "".join(full_answer)


if __name__ == "__main__":
    # 测试调用 流式消息 API
    log("========== 开始测试 ==========")

    # 1. 先创建会话获取真实 session_id
    session_id = create_session()
    log(f"获取到 session_id: {session_id}")

    # 2. 使用真实 session_id 调用流式消息 API
    results = call_api(
        pod_name="hummingbird",
        url_path="",
        request_body={},
        session_id=session_id,
        content="帮我查一下小赵庄东网元的告警"
    )
    # print(f"结果: {results}")
    complete_text = extract_complete_answer(results)

    print("\n" + "=" * 20 + " AI 完整回答 " + "=" * 20)
    print(complete_text)
    print("=" * 53 + "\n")
    log("========== 测试完成 ==========")
