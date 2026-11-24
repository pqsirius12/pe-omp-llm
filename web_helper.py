#!/usr/bin/env python3
"""web_helper.py - collect answers from a web chat (ChatGPT, DeepSeek) by hand,
with the clipboard doing the copying.

  python3 web_helper.py chatgpt
  python3 web_helper.py deepseek

For each of the 30 prompts (15 kernels x plain/hw):
  1. the prompt is put on your clipboard
  2. you open a NEW chat, paste (Cmd+V), send, wait for the full reply
  3. you click the reply's copy button, come back here and press Enter
  4. the reply is saved and its code extracted into variants/
Resumable: answers already saved are skipped. Type 's' + Enter to skip one,
'q' + Enter to stop. macOS only (uses pbcopy/pbpaste).
"""
import datetime, json, os, subprocess, sys
import collect   # build_prompt, kernels, extract_code

ROOT = os.path.dirname(os.path.abspath(__file__))


def clip_set(text):
    subprocess.run(["pbcopy"], input=text.encode(), check=True)


def clip_get():
    return subprocess.run(["pbpaste"], capture_output=True, check=True).stdout.decode(errors="replace")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    site = sys.argv[1].lower()
    model = input(f"Exact model name shown in {site} (e.g. as in its model picker): ").strip()
    if not model:
        sys.exit("A model name is needed for the paper.")
    common = open(os.path.join(ROOT, "kernels", "common.h")).read()
    raw_dir = os.path.join(ROOT, "llm_raw", site)
    os.makedirs(raw_dir, exist_ok=True)
    manifest = open(os.path.join(ROOT, "llm_raw", "manifest.jsonl"), "a")
    items = [(n, p, s) for n, p in collect.kernels().items() for s in ("plain", "hw")]
    for i, (name, path, style) in enumerate(items, 1):
        out_c = os.path.join(ROOT, "variants", name, f"{site}__{style}__s1.c")
        if os.path.exists(out_c):
            continue
        clip_set(collect.build_prompt(style, open(path).read(), common))
        print(f"\n[{i}/{len(items)}] {name} ({style}) - prompt is on your clipboard.")
        print("  NEW chat -> Cmd+V -> send -> wait for the whole reply -> copy it -> press Enter here")
        while True:
            cmd = input("  > ").strip().lower()
            if cmd == "q":
                print("Stopped. Run the same command again to continue."); return
            if cmd == "s":
                print("  skipped"); break
            reply = clip_get()
            code = collect.extract_code(reply)
            if reply.strip() == collect.build_prompt(style, open(path).read(), common).strip():
                print("  The clipboard still holds the prompt - copy the REPLY first, then press Enter.")
                continue
            if not code:
                ans = input("  No C code block found in the copied reply. Save it as a failed answer? (y/n) ")
                if ans.strip().lower() != "y":
                    continue
            open(os.path.join(raw_dir, f"{name}__{style}__s1.txt"), "w").write(reply)
            os.makedirs(os.path.dirname(out_c), exist_ok=True)
            open(out_c, "w").write(code if code else "/* NO CODE BLOCK IN REPLY */\n#error no code\n")
            manifest.write(json.dumps({"model": f"{site}:{model}", "kernel": name, "prompt": style,
                                       "sample": 1, "time": datetime.datetime.now().timestamp(),
                                       "has_code": bool(code), "source": "web"}) + "\n")
            manifest.flush()
            print(f"  saved ({len(reply)} chars, {'code found' if code else 'NO code'})")
            break
    print("\nAll prompts done for", site)


if __name__ == "__main__":
    main()
