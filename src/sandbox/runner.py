"""容器内的常驻 Python 运行器：通过 stdin/stdout 的 JSON 行协议与宿主（Rust）通信。"""
import ast
import contextlib
import io
import json
import signal
import sys
import traceback

_proto_out = sys.stdout  # 协议通道：保存真正的 stdout
_proto_in = sys.stdin
_globals = {"__name__": "__sandbox__"}  # 会话状态：跨多次执行保持
_MAX_CHARS = 100_000  # 输出截断，避免撑爆 LLM 上下文


class _Timeout(BaseException):  # 继承 BaseException，避免被用户代码的 except Exception 吞掉
    pass


def _on_alarm(signum, frame):
    raise _Timeout("execution timed out")


signal.signal(signal.SIGALRM, _on_alarm)


def _send(msg):
    _proto_out.write(json.dumps(msg, ensure_ascii=False) + "\n")
    _proto_out.flush()


def _call_tool(tool_name, args):
    """沙箱内调用宿主 Rust 工具：发出 tool_call，阻塞等待宿主回复。"""
    _send({"type": "tool_call", "name": tool_name, "args": args})
    reply = json.loads(_proto_in.readline())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["result"]


_globals["_call_tool"] = _call_tool


def _clip(text):
    if len(text) <= _MAX_CHARS:
        return text
    return text[:_MAX_CHARS] + f"\n...[truncated {len(text) - _MAX_CHARS} chars]"


def _run(code, timeout):
    out, err = io.StringIO(), io.StringIO()
    value, error = None, None
    try:
        tree = ast.parse(code, "<sandbox>", "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.fix_missing_locations(ast.Expression(tree.body.pop().value))
        signal.setitimer(signal.ITIMER_REAL, timeout)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(tree, "<sandbox>", "exec"), _globals)
            if last is not None:
                v = eval(compile(last, "<sandbox>", "eval"), _globals)
                if v is not None:
                    value = repr(v)
    except BaseException as e:  # noqa: BLE001 - 含超时、SystemExit
        error = {
            "name": type(e).__name__,
            "value": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return {
        "stdout": _clip(out.getvalue()),
        "stderr": _clip(err.getvalue()),
        "value": value,
        "error": error,
    }


def main():
    while True:
        line = _proto_in.readline()
        if not line:
            break
        req = json.loads(line)
        if req["type"] == "exec":
            res = _run(req["code"], req.get("timeout", 30))
            _send({"type": "result", "id": req["id"], **res})
        elif req["type"] == "shutdown":
            break


if __name__ == "__main__":
    main()
