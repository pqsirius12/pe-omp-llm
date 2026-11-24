#!/usr/bin/env python3
"""collect.py - ask an LLM to parallelize every kernel and save its code.

  python3 collect.py prompts                      write prompts/ for web chats
  python3 collect.py gemini --model gemini-3.8-flash [--samples 3]

Two prompt styles per kernel:
  plain : parallelize with OpenMP for speed (no hardware details)
  hw    : same, plus the facts about the Apple M4's 4 P-cores + 6 E-cores
Outputs: variants/<kernel>/<model>__<prompt>__s<k>.c  (code for run.py)
         llm_raw/<model>/<kernel>__<prompt>__s<k>.json (full raw response)
         llm_raw/manifest.jsonl                       (one line per call)
Resumable: a sample whose .c file exists is skipped. Stdlib only.
"""
import argparse, glob, http.client, json, os, re, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")

COMMON = """Parallelize the C program below with OpenMP so that it runs as fast as possible.
Rules:
- Change only the code between the "=== KERNEL BEGIN ===" and "=== KERNEL END ===" markers.
  You may add #include lines at the top of the file.
- Keep main(), the function signature of kernel(), and the output line unchanged.
- The program must give the same result as the serial version.
- It will be compiled with: clang -O2 -fopenmp (OpenMP 5.1) and run with
  OMP_NUM_THREADS set to each value from 1 to 10.
"""
HW = """Target machine: Apple M4 (MacBook Air, macOS). Its 10 CPU cores are not identical:
4 performance cores (P-cores) and 6 efficiency cores (E-cores), which are slower.
The operating system decides which core each thread runs on; threads cannot be
pinned to specific cores.
"""
TAIL = """Reply with the complete modified program in a single ```c code block.

The program includes "common.h", shown first for reference (do not change it):
```c
{common}
```

Program to parallelize:
```c
{program}
```
"""


def build_prompt(style, program, common):
    head = COMMON + ("\n" + HW if style == "hw" else "")
    return head + "\n" + TAIL.format(common=common, program=program)


def kernels():
    return {os.path.basename(p)[4:-2]: p
            for p in sorted(glob.glob(os.path.join(ROOT, "kernels", "k[0-9][0-9]_*.c")))}


def extract_code(text):
    """Largest ```c block in the reply; None if there is none."""
    blocks = re.findall(r"```(?:c|C|cpp)?[ \t]*\n(.*?)```", text, re.S)
    blocks = [b for b in blocks if "main" in b and "kernel" in b]
    return max(blocks, key=len) if blocks else None


class QuotaExhausted(Exception):
    """The model's daily request quota is used up (resets midnight Pacific)."""


def gemini_call(model, prompt, key, timeout=600):
    """One generateContent request; returns the parsed JSON. Retries on 429/5xx."""
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(f"{API}/models/{model}:generateContent", data=body,
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": key})
    wait = int(os.environ.get("RETRY_WAIT", "20"))
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:600]
            # a used-up quota will not recover within this run: stop this model
            if e.code == 429 and "quota" in msg.lower():
                raise QuotaExhausted(msg)
            if e.code in (429, 500, 502, 503, 504):
                print(f"    HTTP {e.code}, retrying in {wait}s: {msg[:120]}", flush=True)
                time.sleep(wait); wait = min(wait * 2, 600)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        # dropped connections, resets and timeouts: retry, never crash the run
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            print(f"    network error, retrying in {wait}s: {e}", flush=True)
            time.sleep(wait); wait = min(wait * 2, 600)
    raise RuntimeError("gave up after 8 attempts")


def reply_text(resp):
    parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def cmd_prompts(a):
    common = open(os.path.join(ROOT, "kernels", "common.h")).read()
    os.makedirs(os.path.join(ROOT, "prompts"), exist_ok=True)
    for name, path in kernels().items():
        for style in ("plain", "hw"):
            p = build_prompt(style, open(path).read(), common)
            open(os.path.join(ROOT, "prompts", f"{name}__{style}.txt"), "w").write(p)
    print("wrote", len(kernels()) * 2, "prompt files to prompts/")


def cmd_gemini(a):
    key = open(os.path.join(ROOT, ".gemini_key")).read().strip()
    common = open(os.path.join(ROOT, "kernels", "common.h")).read()
    tag = a.model.replace("/", "_").replace(".", "_")
    raw_dir = os.path.join(ROOT, "llm_raw", tag)
    os.makedirs(raw_dir, exist_ok=True)
    manifest = open(os.path.join(ROOT, "llm_raw", "manifest.jsonl"), "a")
    only = set(a.kernels.split(",")) if a.kernels else None
    for name, path in kernels().items():
        if only and name not in only:
            continue
        for style in ("plain", "hw"):
            for s in range(1, a.samples + 1):
                label = f"{tag}__{style}__s{s}"
                out_c = os.path.join(ROOT, "variants", name, label + ".c")
                if os.path.exists(out_c):
                    continue                          # resume: already collected
                prompt = build_prompt(style, open(path).read(), common)
                t0 = time.time()
                try:
                    resp = gemini_call(a.model, prompt, key)
                except QuotaExhausted as e:
                    print(f"  QUOTA USED UP for {a.model}: stopping this model. Run the same"
                          f" command again after the daily reset to continue.\n  {str(e)[:300]}")
                    return
                except RuntimeError as e:
                    print(f"  {name} {style} s{s}: FAILED {e}", flush=True)
                    manifest.write(json.dumps({"model": a.model, "kernel": name, "prompt": style,
                                               "sample": s, "error": str(e), "time": t0}) + "\n")
                    manifest.flush()
                    continue
                json.dump(resp, open(os.path.join(raw_dir, f"{name}__{style}__s{s}.json"), "w"), indent=1)
                text = reply_text(resp)
                code = extract_code(text)
                os.makedirs(os.path.dirname(out_c), exist_ok=True)
                # no code block = a failed answer; record it as a file that will not compile
                open(out_c, "w").write(code if code else "/* NO CODE BLOCK IN REPLY */\n#error no code\n")
                manifest.write(json.dumps({
                    "model": a.model, "model_version": resp.get("modelVersion"),
                    "kernel": name, "prompt": style, "sample": s, "time": t0,
                    "seconds": round(time.time() - t0, 1), "has_code": bool(code),
                    "finish": resp.get("candidates", [{}])[0].get("finishReason"),
                    "usage": resp.get("usageMetadata")}) + "\n")
                manifest.flush()
                print(f"  {name:10s} {style:5s} s{s}: {'ok' if code else 'NO CODE'} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                time.sleep(a.pause)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts")
    g = sub.add_parser("gemini")
    g.add_argument("--model", required=True)
    g.add_argument("--samples", type=int, default=3)
    g.add_argument("--kernels", default="")
    g.add_argument("--pause", type=float, default=4.0, help="seconds between calls")
    a = p.parse_args()
    cmd_prompts(a) if a.cmd == "prompts" else cmd_gemini(a)


if __name__ == "__main__":
    main()
