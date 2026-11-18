# The two answer prompts of the owner's `data/host-pass7/e1_qa.py` (the owner, 2026-10-26; git-ignored, so
# not in the repo), copied verbatim as Python source. That file's sha256:
#   47eae4614b8c62fc81c8e16a9930e80cad38c89078142820122f6a3148fa69ab
# Transcribed from Bosmang's quote of the file in MEM2-227 (the file itself is not in the sandbox).
# tests/bench_t0/test_e1.py loads this file with exec (never imports data/) and pins its own sha256;
# bench/e1.py's ANSWER and ANSWER_NONE must equal these byte for byte. Do not reformat.
# fmt: off
ANSWER = ("I will give you several history chats between you and a user. Please answer the question based on the "
          "relevant chat history. If the history does not contain the answer, say that you don't know.\n\n"
          "History Chats:\n{ctx}\n\nCurrent Date: {date}\nQuestion: {q}\nAnswer:")
ANSWER_NONE = "Current Date: {date}\nQuestion: {q}\nAnswer (say you don't know if you can't know):"
# fmt: on
