"""ds_collect.py - collect answers from the DeepSeek API (pinned, reproducible model).

API (checked against api-docs.deepseek.com on 23 Nov 2026):
  POST https://api.deepseek.com/chat/completions   header  Authorization: Bearer <key>
  OpenAI-style body {"model": ..., "messages": [...], "stream": false}
  model "deepseek-flash" is served by DeepSeek-V4.1-Flash-1017.
  Thinking is turned ON by {"thinking": {"type": "enabled"}}; we send no thinking
  field (the web runs had DeepThink off) and `test` reports whether any reasoning
  text comes back anyway, so the setting is verified, not assumed.

Commands
  python3 ds_collect.py test [--thinking disabled]   one tiny request; prints model, fields, usage
  python3 ds_collect.py collect [--samples 3]    plain + hw prompts for all 15 kernels
Outputs: variants/<k>/deepseek-api__<style>__s<n>.c, llm_raw/deepseek-api/*.json,
         llm_raw/manifest.jsonl (model "deepseek-flash", model_version from the response)
Resumable: answers already on disk are skipped. The key is read from .deepseek_key
and never printed.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request, http.client
import collect as C

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
TAG = "deepseek-api"


THINKING = None   # set from --thinking: None = send no field, "disabled" = {"type": "disabled"}


def call(prompt, key, timeout=600, tries=6):
    req_body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}
    if THINKING:
        req_body["thinking"] = {"type": THINKING}
    body = json.dumps(req_body).encode()
    wait = 20
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (400, 401, 402, 403):     # bad request / key / balance: retrying cannot help
                sys.exit(f"HTTP {e.code}: {msg}")
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                print(f"    HTTP {e.code}, retrying in {wait}s", flush=True); time.sleep(wait); wait = min(wait * 2, 300); continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            if attempt < tries - 1:
                print(f"    network error {e}, retrying in {wait}s", flush=True); time.sleep(wait); wait = min(wait * 2, 300); continue
            raise RuntimeError(f"network: {e}")
    raise RuntimeError("gave up")


def key():
    k = open(os.path.join(ROOT, ".deepseek_key")).read().strip()
    if not k.startswith("sk-"):
        sys.exit(".deepseek_key does not look like a DeepSeek key")
    return k


def cmd_test(a):
    r = call("Reply with the single word OK.", key(), timeout=120, tries=2)
    msg = r["choices"][0]["message"]
    print("model field     :", r.get("model"))
    print("message fields  :", sorted(msg.keys()))
    print("answer text     :", repr((msg.get("content") or "")[:60]))
    print("reasoning text  :", len(msg.get("reasoning_content") or ""), "characters (0 = thinking off)")
    print("usage           :", r.get("usage"))
    print("finish_reason   :", r["choices"][0].get("finish_reason"))


def cmd_collect(a):
    k = key()
    progs = {kk: open(p).read() for kk, p in C.kernels().items()}
    common = open(os.path.join(ROOT, "kernels", "common.h")).read()
    rawdir = os.path.join(ROOT, "llm_raw", TAG); os.makedirs(rawdir, exist_ok=True)
    done = 0
    for kern in sorted(progs):
        for style in ("plain", "hw"):
            prompt = C.build_prompt(style, progs[kern], common)
            for s in range(1, a.samples + 1):
                stem = f"{TAG}__{style}__s{s}"
                out = os.path.join(ROOT, "variants", kern, stem + ".c")
                rawp = os.path.join(rawdir, f"{kern}__{stem}.json")
                if os.path.exists(rawp):
                    continue
                t0 = time.time()
                try:
                    r = call(prompt, k)
                except RuntimeError as e:
                    print(f"  {kern:10s} {style:5s} s{s}: FAILED {e}", flush=True); continue
                msg = r["choices"][0]["message"]
                text = msg.get("content") or ""
                code = C.extract_code(text)
                json.dump({"prompt": prompt, "response": r}, open(rawp, "w"))
                if code:
                    open(out, "w").write(code)
                with open(os.path.join(ROOT, "llm_raw", "manifest.jsonl"), "a") as f:
                    f.write(json.dumps({"model": MODEL, "model_version": r.get("model"), "kernel": kern,
                                        "prompt": style, "sample": s, "time": time.time(),
                                        "seconds": round(time.time() - t0, 1), "has_code": bool(code),
                                        "finish": r["choices"][0].get("finish_reason"),
                                        "reasoning_chars": len(msg.get("reasoning_content") or ""),
                                        "thinking_param": THINKING or "not sent",
                                        "usage": r.get("usage")}) + "\n")
                done += 1
                print(f"  {kern:10s} {style:5s} s{s} code={'yes' if code else 'NO'} {time.time() - t0:.0f}s", flush=True)
                time.sleep(a.pause)
    print("new answers:", done)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    c = sub.add_parser("collect"); c.add_argument("--samples", type=int, default=3)
    c.add_argument("--pause", type=float, default=1)
    for sp in (sub.choices["test"], c):
        sp.add_argument("--thinking", choices=["default", "disabled"], default="default",
                        help="default: send no thinking field; disabled: send {\"type\": \"disabled\"}")
    a = p.parse_args()
    global THINKING
    THINKING = None if a.thinking == "default" else a.thinking
    {"test": cmd_test, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
