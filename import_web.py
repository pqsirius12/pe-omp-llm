"""import_web.py - turn browser-saved chatbot replies into benchmark variants.

Reads llm_raw/web/pe__chatgpt__<kernel>__<style>.json,
llm_raw/web/pe__deepseek__bundle30.json and llm_raw/web/pe__claude__bundle30.json
(claude.ai incognito chats, collected by copy/paste through the UI) and writes
  variants/<kernel>/<tag>__<style>__s1.c       (only when the reply has code)
  llm_raw/manifest.jsonl                       (one line per reply, code or not)
Replies with no complete program get a manifest line with has_code=false, so
they count as failures in the analysis instead of silently disappearing.
Safe to re-run: it rewrites the .c files and skips manifest lines already present.
"""
import glob, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "llm_raw", "web")
MAN = os.path.join(ROOT, "llm_raw", "manifest.jsonl")
MODELS = {
    "chatgpt": ("chatgpt-web", "chatgpt:web-plus-default-think-off"),
    "deepseek": ("deepseek-web", "deepseek:web-default-deepthink-off-search-off"),
    "claude": ("claude-web", "claude:web-opus-5.5-default-effort-incognito"),
}


def load_records():
    recs = []
    for path in sorted(glob.glob(os.path.join(WEB, "pe__chatgpt__*.json"))):
        if "(" in os.path.basename(path):       # browser duplicate "(1)" copies
            continue
        recs.append(json.load(open(path)))
    for name in ("pe__deepseek__bundle30.json", "pe__claude__bundle30.json"):
        bundle = os.path.join(WEB, name)
        if os.path.exists(bundle):
            recs += json.load(open(bundle))
    # feedback round (llm_raw/web/fb/): bundles and single-record files, possibly
    # with browser duplicates "(1)"; keep the most recently saved copy per answer
    fb = {}
    for path in sorted(glob.glob(os.path.join(WEB, "fb", "*.json"))):
        data = json.load(open(path))
        for r in (data if isinstance(data, list) else [data]):
            key = (r["site"], r["kernel"], r["style"])
            if key not in fb or r.get("saved", "") > fb[key].get("saved", ""):
                fb[key] = r
    return recs + list(fb.values())


def main():
    done = set()
    if os.path.exists(MAN):
        for line in open(MAN):
            m = json.loads(line)
            done.add((m["model"], m["kernel"], m["prompt"], m.get("sample")))
    seen, written, nocode = set(), 0, []
    with open(MAN, "a") as man:
        for r in load_records():
            tag, model = MODELS[r["site"]]
            key = (model, r["kernel"], r["style"], 1)
            if key in seen:
                raise SystemExit("duplicate reply for %s" % (key,))
            seen.add(key)
            code = r.get("code")
            has = bool(code and "main" in code and "kernel" in code)
            if has:
                out = os.path.join(ROOT, "variants", r["kernel"],
                                   "%s__%s__s1.c" % (tag, r["style"]))
                with open(out, "w") as f:
                    f.write(code if code.endswith("\n") else code + "\n")
                written += 1
            else:
                nocode.append("%s/%s/%s" % (r["site"], r["kernel"], r["style"]))
            if key not in done:
                man.write(json.dumps({
                    "model": model, "model_version": model, "kernel": r["kernel"],
                    "prompt": r["style"], "sample": 1, "has_code": has,
                    "source": "web", "url": r.get("url"), "saved": r.get("saved"),
                    "reply_chars": len(r.get("reply") or ""),
                    "note": r.get("model") or ""}) + "\n")
    print("replies:", len(seen), "variants written:", written)
    print("no complete program:", ", ".join(nocode) or "none")


if __name__ == "__main__":
    main()
