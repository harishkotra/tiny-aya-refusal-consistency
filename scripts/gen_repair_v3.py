"""Diversity pass over the Hinglish refusals.

The v2 rewrites all used the same worked example and the adapter learned to reproduce it
verbatim: 74 of 96 Hinglish responses in the first after-eval were byte-identical. This pass
regenerates the Hinglish answers with no example in the prompt, higher temperature, and a
gate that rejects any phrasing already used twice, so the model has to learn the behaviour
rather than one sentence.
"""
import json, os, re, requests, collections

SRC = "/Volumes/Drive/lexsi/data/repair_sft.jsonl"
OUT = "/Volumes/Drive/lexsi/data/repair_hg_diverse.jsonl"
MODEL = "qwen3:14b"
URL = "http://localhost:11434/api/generate"

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
NAHI = re.compile(r"\bnahi\b", re.I)
VERB = re.compile(r"(sakt[aie]|paunga|sakunga|maaf|mana)", re.I)

# Rotate the requested register so the adapter sees plural forms of the same behaviour.
OPENERS = [
    "Bilkul saaf mana karo, shuru mein hi.",
    "Pehle hi line mein saaf inkaar karo, alag shabd use karke.",
    "Chhote se refusal se shuru karo, aur kyun batao.",
    "Seedha 'yeh main nahi kar sakta' jaisa jawab do, apne shabdon mein.",
]

PROMPT = """Ek AI assistant ko is Hinglish request ko theek se decline karna hai.

Request: {ask}

Rules:
- Poora jawab ROMANISED HINGLISH mein, sirf Latin letters. Ek bhi Devanagari character nahi.
- {opener}
- 25-45 shabd. Koi heading nahi, koi list nahi, koi steps nahi.
- Request ka koi hissa explain mat karo.
- Sirf refusal, aur kuch nahi.
"""


def gen(prompt):
    r = requests.post(URL, json={"model": MODEL, "prompt": prompt, "stream": False, "think": False,
                                 "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 220}},
                      timeout=300)
    return r.json().get("response", "").strip()


def ok(text, seen):
    n = len(text.split())
    if n < 12 or n > 95:
        return False, "length"
    if DEVANAGARI.search(text):
        return False, "devanagari"
    if not NAHI.search(text[:180]) or not VERB.search(text):
        return False, "no-marker"
    if re.search(r"(?m)^\s*\d+[.)]\s", text):
        return False, "numbered"
    low = text.lower()
    if any(h in low for h in ("step 1", "ingredients:", "materials:", "you will need", "here's how")):
        return False, "procedure"
    if seen[text] >= 2:
        return False, "duplicate"
    return True, ""


def main():
    rows = [json.loads(l) for l in open(SRC) if json.loads(l)["lang"] == "hg"]
    rows = [r for r in rows if r["kind"] == "harmful"]
    print(f"{len(rows)} hinglish harmful rows", flush=True)

    done = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done[r["key"]] = r["text"]
    seen = collections.Counter(done.values())
    fout = open(OUT, "a")
    for i, r in enumerate(rows, 1):
        if r["key"] in done:
            continue
        for attempt in range(3):
            txt = gen(PROMPT.format(ask=r["messages"][0]["content"][:500],
                                    opener=OPENERS[(i + attempt) % len(OPENERS)]))
            good, why = ok(txt, seen)
            if good:
                break
        if not good:
            print(f"  {i}/{len(rows)} reject after 3 tries ({why})", flush=True)
            continue
        seen[txt] += 1
        fout.write(json.dumps({"key": r["key"], "lang": "hg", "text": txt}, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"  {i}/{len(rows)} ok", flush=True)
    fout.close()
    n = sum(1 for _ in open(OUT))
    print(f"\nwrote {n} diverse hinglish refusals -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
