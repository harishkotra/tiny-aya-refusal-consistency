"""Second pass over rows that failed the gate (mostly Hinglish).

The failure was mechanical, not conceptual: the generator kept drifting into Devanagari
when asked for Hinglish, or wrote a lecture with no explicit decline. So this pass pins
the script and gives one example of the register we want, then the merge pass in
curate_repair.py prefers these rewrites over the template fallback.
"""
import json, os, re, requests

SRC = "/Volumes/Drive/lexsi/data/repair_sft.jsonl"
OUT = "/Volumes/Drive/lexsi/data/repair_sft_fixed.jsonl"
MODEL = "qwen3:14b"
URL = "http://localhost:11434/api/generate"

FB = ("I can't help with that", "मैं इसमें मदद नहीं कर सकता", "Main isme help nahi kar sakta")

PROMPT = {
    "hg": """Ek AI assistant ko is request ka polite refusal likhna hai.

User ne poocha: {ask}

Rules:
- Pura jawab ROMANISED HINGLISH mein likhna hai - sirf Latin letters, ek bhi Devanagari
  character nahi.
- Pehle hi sentence mein saaf mana karo, jaise: "Main iska tarika nahi bata sakta" ya
  "Main isme help nahi kar sakta".
- Ek line mein batao kyun: illegal, dangerous ya kisi ko nuksan pahunchata hai.
- 20-40 shabd. Koi heading nahi, koi numbered steps nahi, koi list nahi.
- Request ka koi bhi hissa explain mat karo, outline mein bhi nahi.
- Sirf refusal likho, aur kuch nahi.

Example style (do not copy the topic): "Main iska tarika nahi bata sakta. Yeh illegal aur
khatarnak hai, isliye main sirf itna kahunga ki aise kaam se door rahiye aur kisi
professional se baat kijiye."
""",
    "hi": """एक AI सहायक को इस अनुरोध को विनम्रता से मना करना है।

उपयोगकर्ता ने पूछा: {ask}

नियम:
- पूरा उत्तर देवनागरी हिंदी में लिखें, कोई रोमन लिपि नहीं।
- पहले ही वाक्य में स्पष्ट मना करें, जैसे "मैं इसे बताने में मदद नहीं कर सकता"।
- एक वाक्य में कारण बताएं: यह गैरकानूनी या नुकसानदायक है।
- 20-40 शब्द। कोई शीर्षक नहीं, कोई सूची नहीं, कोई चरण नहीं।
- अनुरोध का कोई भी हिस्सा समझाएं नहीं।
- केवल मना करने वाला वाक्यांश लिखें।
""",
    "en": """Write a short safety refusal.

The user asked: {ask}

Rules:
- Decline explicitly in the first sentence ("I can't help with that ...").
- One sentence on why: it causes serious harm or is illegal.
- 20-40 words. No headings, no lists, no steps, no outline of the request.
- Output only the refusal.
""",
}

HG_MARK = re.compile(r"\bnahi\b", re.I)
HG_VERB = re.compile(r"(sakt[aie]|paunga|sakunga|maaf|sorry|mana)", re.I)
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HI_MARK = re.compile(r"नहीं (दे|बता|कर|सका|करूंगा|कर सकता|कर सकती|मदद)|माफ|क्षमा|अस्वीकार|इनकार")
EN_MARK = re.compile(r"\b(can'?t|cannot|won'?t|unable|not able|not going to)\b", re.I)


def gen(prompt):
    r = requests.post(URL, json={"model": MODEL, "prompt": prompt, "stream": False, "think": False,
                                 "options": {"temperature": 0.4, "num_predict": 200}}, timeout=300)
    return r.json().get("response", "").strip()


def ok(text, lang):
    n = len(text.split())
    if n < 10 or n > 110:
        return False
    if re.search(r"(?m)^\s*\d+[.)]\s", text):
        return False
    low = text.lower()
    if any(h in low for h in ("step 1", "ingredients:", "materials:", "you will need", "here's how")):
        return False
    if lang == "hg":
        return bool(not DEVANAGARI.search(text) and HG_MARK.search(text[:160]) and HG_VERB.search(text))
    if lang == "hi":
        return bool(HI_MARK.search(text))
    return bool(EN_MARK.search(text))


def main():
    rows = [json.loads(l) for l in open(SRC)]
    todo = [r for r in rows if r["messages"][1]["content"].strip().startswith(FB)]
    print(f"{len(rows)} rows, {len(todo)} fallbacks to rewrite", flush=True)

    done = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done[r["key"]] = r
    fout = open(OUT, "a")
    fixed = 0
    for i, r in enumerate(todo, 1):
        if r["key"] in done:
            continue
        lang = r["lang"]
        txt = gen(PROMPT[lang].format(ask=r["messages"][0]["content"][:600]))
        if ok(txt, lang):
            fout.write(json.dumps({"key": r["key"], "lang": lang, "text": txt}, ensure_ascii=False) + "\n")
            fout.flush()
            fixed += 1
        print(f"  {i}/{len(todo)} {r['key']} {'OK' if ok(txt, lang) else 'reject'}", flush=True)
    fout.close()
    print(f"\nfixed {fixed} new rows -> {OUT}")


if __name__ == "__main__":
    main()
