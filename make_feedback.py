"""make_feedback.py - round-2 prompts: feedback round (RQ4) and explicit-request style.

feedback ("fb"): for every CORRECT answer to the plain prompt, a fresh chat gets the
  same plain prompt plus that answer's code and its MEASURED times (serial, 1/4/10
  threads, median of 5 runs). Measurement only: no hint about schedules or cores,
  so any change comes from the model's own reading of the numbers.
explicit ("hwx"): the hw prompt plus one sentence asking for code that copes with
  the slower E-cores - separates "did not know" from "knew but did not act".

Commands
  python3 make_feedback.py prompts                 write prompts/fb/*.txt, prompts/hwx/*.txt
  python3 make_feedback.py gemini --model gemma-4-31b-it [--pause 5]
        collect Gemma answers for fb (one per original sample) and hwx (3 samples)
Inputs: feedback_meas.json (made by analysis/analyze.py; times of correct plain answers)
Outputs: variants/<k>/<tag>__fb__s<k>.c, variants/<k>/<tag>__hwx__s<k>.c,
         llm_raw/<tag>/*.json, llm_raw/manifest.jsonl (prompt = "fb" / "hwx")
"""
import argparse, json, os, sys, time
import collect as C

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPLICIT = ("Write the code so that it performs well on this machine despite the "
            "slower E-cores.\n")
FB_HEAD = ("\nA previous answer to this request is shown below. It compiles and gives the "
           "correct result. It was measured on the target machine (median of 5 runs, "
           "wall-clock time of kernel() only):\n"
           "- serial version: {ser:.3f} s\n"
           "- previous answer: {t1:.3f} s with OMP_NUM_THREADS=1, {t4:.3f} s with 4, "
           "{t10:.3f} s with 10\n"
           "Improve the performance of the previous answer, keeping all the rules above. "
           "Reply with the complete modified program in a single ```c code block.\n\n"
           "Previous answer:\n```c\n{code}```\n")
VTAG = {"gemma": "gemma-4-31b-it", "chatgpt": "chatgpt-web",
        "deepseek": "deepseek-web", "claude": "claude-web"}


def fb_prompt(kernel, code, m, progs, common):
    base = C.build_prompt("plain", progs[kernel], common)
    code = code if code.endswith("\n") else code + "\n"
    return base + FB_HEAD.format(ser=m["serial"], t1=m["T1"], t4=m["T4"], t10=m["T10"], code=code)


def hwx_prompt(kernel, progs, common):
    head = C.COMMON + "\n" + C.HW + EXPLICIT
    return head + "\n" + C.TAIL.format(common=common, program=progs[kernel])


def load():
    progs = {k: open(p).read() for k, p in C.kernels().items()}
    common = open(os.path.join(ROOT, "kernels", "common.h")).read()
    meas = json.load(open(os.path.join(ROOT, "feedback_meas.json")))
    return progs, common, meas


def cmd_prompts(a):
    progs, common, meas = load()
    for d in ("fb", "hwx"):
        os.makedirs(os.path.join(ROOT, "prompts", d), exist_ok=True)
    n = 0
    for m in meas:
        src = os.path.join(ROOT, "variants", m["kernel"], m["variant"] + ".c")
        p = fb_prompt(m["kernel"], open(src).read(), m, progs, common)
        name = f"{m['model']}__{m['kernel']}__s{m['sample']}.txt"
        open(os.path.join(ROOT, "prompts", "fb", name), "w").write(p); n += 1
    for k in progs:
        open(os.path.join(ROOT, "prompts", "hwx", f"{k}.txt"), "w").write(hwx_prompt(k, progs, common))
    print(f"fb prompts: {n}, hwx prompts: {len(progs)}")


def save(model, tag, kernel, style, sample, prompt, resp, secs):
    text = C.reply_text(resp)
    code = C.extract_code(text) if text else None
    rawdir = os.path.join(ROOT, "llm_raw", tag); os.makedirs(rawdir, exist_ok=True)
    stem = f"{tag}__{style}__s{sample}"
    json.dump({"prompt": prompt, "response": resp}, open(os.path.join(rawdir, f"{kernel}__{stem}.json"), "w"))
    if code:
        open(os.path.join(ROOT, "variants", kernel, stem + ".c"), "w").write(code)
    with open(os.path.join(ROOT, "llm_raw", "manifest.jsonl"), "a") as f:
        f.write(json.dumps({"model": model, "model_version": resp.get("modelVersion", model),
                            "kernel": kernel, "prompt": style, "sample": sample,
                            "time": time.time(), "seconds": round(secs, 1), "has_code": bool(code),
                            "finish": (resp.get("candidates") or [{}])[0].get("finishReason"),
                            "usage": resp.get("usageMetadata")}) + "\n")
    return bool(code)


def cmd_gemini(a):
    key = open(os.path.join(ROOT, ".gemini_key")).read().strip()
    tag = a.model.replace(".", "_")
    progs, common, meas = load()
    jobs = []
    for m in meas:
        if m["model"] == "gemma" and a.model.startswith("gemma"):
            src = os.path.join(ROOT, "variants", m["kernel"], m["variant"] + ".c")
            jobs.append((m["kernel"], "fb", m["sample"], fb_prompt(m["kernel"], open(src).read(), m, progs, common)))
    for k in sorted(progs):
        for s in range(1, a.hwx_samples + 1):
            jobs.append((k, "hwx", s, hwx_prompt(k, progs, common)))
    for k, style, s, prompt in jobs:
        if os.path.exists(os.path.join(ROOT, "variants", k, f"{tag}__{style}__s{s}.c")):
            continue                              # resumable
        t0 = time.time()
        try:
            resp = C.gemini_call(a.model, prompt, key)
        except C.QuotaExhausted as e:
            sys.exit(f"quota exhausted: {str(e)[:200]}")
        except RuntimeError as e:
            print(f"  {k} {style} s{s}: FAILED {e}"); continue
        ok = save(a.model, tag, k, style, s, prompt, resp, time.time() - t0)
        print(f"  {k:10s} {style:3s} s{s} code={'yes' if ok else 'NO'} {time.time() - t0:.0f}s", flush=True)
        time.sleep(a.pause)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts")
    g = sub.add_parser("gemini"); g.add_argument("--model", default="gemma-4-31b-it")
    g.add_argument("--pause", type=float, default=5)
    g.add_argument("--hwx-samples", type=int, default=3, help="samples per kernel for hwx (default 3)")
    a = p.parse_args()
    {"prompts": cmd_prompts, "gemini": cmd_gemini}[a.cmd](a)


if __name__ == "__main__":
    main()
