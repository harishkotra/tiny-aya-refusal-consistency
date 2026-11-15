"""Bring the dataset cards up to date with the final protocol and cross-link the artefacts."""
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
NS = "harishkotra"
COLL = f"https://huggingface.co/collections/{NS}/tiny-aya-global-refusal-consistency-across-scripts-6af9007d3ce7e09c334db33a"
CODE = "https://github.com/harishkotra/tiny-aya-refusal-consistency"

LINKS = f"""
## Where the rest of it lives

- code + article: {CODE}
- collection (adapter, prompts, SFT set, eval runs): {COLL}
"""


def card(repo):
    p = hf_hub_download(f"{NS}/{repo}", "README.md", repo_type="dataset")
    return Path(p).read_text()


# 1. eval card: describe the final protocol, not just the token settings
t = card("tiny-aya-refusal-consistency-eval")
old = """Decoding: greedy, 256 new tokens, `eos_token_id` includes both `<|END_OF_TURN_TOKEN|>` and
`<|END_RESPONSE|>`."""
new = """Decoding (identical in both arms): greedy, 256 new tokens, `repetition_penalty=1.10`,
`eos_token_id` includes both `<|END_OF_TURN_TOKEN|>` and `<|END_RESPONSE|>`. The penalty is applied
to both arms because it is a decoder property — applied to one arm it would measure the decoder
rather than the repair. See `PROTOCOL.md`."""
assert old in t, "eval card anchor missing"
t = t.replace(old, new)
if "Where the rest of it lives" not in t:
    t = t.rstrip() + "\n" + LINKS
api.upload_file(path_or_fileobj=t.encode(), path_in_repo="README.md",
                repo_id=f"{NS}/tiny-aya-refusal-consistency-eval", repo_type="dataset",
                commit_message="document the final decoding protocol and cross-link the collection")

# 2. prompts card: cross-link, and state what the set is for
t = card("tiny-aya-refusal-consistency-prompts")
if "Where the rest of it lives" not in t:
    t = t.rstrip() + "\n" + LINKS
api.upload_file(path_or_fileobj=t.encode(), path_in_repo="README.md",
                repo_id=f"{NS}/tiny-aya-refusal-consistency-prompts", repo_type="dataset",
                commit_message="cross-link the collection and code")

# 3. sft card: cross-link
t = card("tiny-aya-refusal-consistency-sft")
if "Where the rest of it lives" not in t:
    t = t.rstrip() + "\n" + LINKS
api.upload_file(path_or_fileobj=t.encode(), path_in_repo="README.md",
                repo_id=f"{NS}/tiny-aya-refusal-consistency-sft", repo_type="dataset",
                commit_message="cross-link the collection and code")

for r in ("tiny-aya-refusal-consistency-eval", "tiny-aya-refusal-consistency-prompts",
          "tiny-aya-refusal-consistency-sft"):
    tt = card(r)
    print(r, "| protocol:", "repetition_penalty=1.10" in tt, "| links:", CODE in tt)
