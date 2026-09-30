#!/usr/bin/env python3
"""vid: local video pipeline for the edit-video and carousel skills.

Heavy work (speech recognition, mistake detection, cutting, HyperFrames
composition, face detection) runs locally. The model reads only compact
text files (transcript.txt, mistakes.json, cut.txt, reports) and writes small
JSON decisions (cuts.json, composition.json, slides.json). Flow: `vid edit PATH`,
see .claude/skills/edit-video/SKILL.md. Colours and fonts: brand/theme.css.

Work directory (one per source video):
  project.json          source path and stream info
  words.json            GigaAM/Whisper segments with word timestamps
  transcript.txt        one line per segment: [start] text        <- model reads
  takes.json            per-segment finished/mumble/stutter flags        <- model reads
  mistakes.json         word-level cut candidates                          <- model reads
  cuts.json             approved cuts (drop/keep takes, source-time ranges) <- model writes
  cut.json              kept takes and words on the cut timeline (out/cut.mp4)
  cut.txt               one line per kept take on the cut timeline         <- model reads
  composition.json      template, pip spans, stage beats, caption emphasis/fixes <- model writes
  out/                  final.mp4 + .srt, report, contact sheet
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BRAND = ROOT / "brand"  # theme.css (colours, fonts), fonts/, brand.json (handle)
YUNET = HERE / "models" / "face_detection_yunet_2023mar.onnx"
HYPERFRAMES = HERE / "hyperframes"
# HyperFrames project: compositions/*.html are the frames (vid compose, vid carousel)
HF_VERSION = "0.8.80"  # pinned; tools/hyperframes/package.json must match
os.environ.setdefault("HYPERFRAMES_NO_TELEMETRY", "1")  # no anonymous usage stats from users' machines

# Russian mat roots and common English swears, matched at the start of a normalized word.
PROFANITY = re.compile(r"^(бля|хуй|хуе|хуё|хуя|пизд|еба|ёба|ебл|ёбн|уеб|заеб|сука|суки|fuck|shit|bitch)")


# ---------- helpers ----------

def die(msg):
    print(f"vid: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        die(f"command failed: {' '.join(map(str, cmd[:6]))} ...\n{r.stderr[-2000:]}")
    return r.stdout


def load(p):
    return json.loads(Path(p).read_text())


def save(p, obj, indent=None):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=indent))


def ts(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}"


def local_path(p):
    """A path as given by the user: Windows paths (C:\\Users\\...) are mapped into WSL (/mnt/c/...)."""
    p = str(p).strip().strip('"').strip("'")
    if re.match(r"^[A-Za-z]:[\\/]", p) and shutil.which("wslpath"):
        p = subprocess.run(["wslpath", "-u", p], capture_output=True, text=True).stdout.strip() or p
    return Path(p).expanduser().resolve()


def default_work(src):
    """Work dir for a source video: videos/<name>.work in the project, wherever the source is."""
    return ROOT / "videos" / (src.stem + ".work")


def project(work):
    work = Path(work)
    if not (work / "project.json").exists():
        die(f"{work} is not a work dir; run: vid.py init VIDEO -w {work}")
    return work, load(work / "project.json")


def probe(path):
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not v:
        die(f"no video stream in {path}")
    num, den = map(int, v.get("avg_frame_rate", "30/1").split("/"))
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        r = int(sd.get("rotation", 0) or 0)
        if r:
            rot = r
    if abs(rot) in (90, 270):
        w, h = h, w
    return {
        "width": w, "height": h, "fps": round(num / den, 3) if den else 30.0,
        "duration": float(info["format"]["duration"]),
        "audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


# ---------- init ----------

def cmd_init(a):
    src = local_path(a.video)
    if not src.exists():
        die(f"no such file: {src}")
    work = Path(a.work) if a.work else default_work(src)
    work.mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    p = {"source": str(src), **probe(src)}
    save(work / "project.json", p, 2)
    print(f"{work}  {p['width']}x{p['height']} {p['fps']}fps {ts(p['duration'])} audio={p['audio']}")


# ---------- transcribe ----------

LANGS = ("ru", "en")  # the only languages the footage is in; auto picks between them


def detect_language(src):
    """ru or en, via Whisper base on the speech (VAD) of up to 4 x 30 s windows spread over the
    file (GigaAM is Russian-only, so the engine choice needs this before transcribing). The first
    30 s alone failed: leading noise/silence gave 'nn' for clear Russian."""
    from faster_whisper import WhisperModel, decode_audio
    audio = decode_audio(src, sampling_rate=16000)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    step = 30 * 16000
    starts = sorted({min(max(0, len(audio) - step), round(len(audio) * f)) for f in (0.1, 0.35, 0.6, 0.85)})
    score = dict.fromkeys(LANGS, 0.0)
    for s in starts:
        try:
            _, _, probs = model.detect_language(audio[s:s + step], vad_filter=True)
        except Exception:  # a window with no speech after VAD
            continue
        for lang, p in probs:
            if lang in score:
                score[lang] += p
    return max(score, key=score.get) if any(score.values()) else LANGS[0]


def transcribe_whisper(src, lang, model_name, prompt):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    segs, info = model.transcribe(src, language=lang, word_timestamps=True, vad_filter=True,
                                  initial_prompt=prompt or None)
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "words": [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2),
                       "p": round(w.probability, 2)} for w in (s.words or [])]} for s in segs]
    return info.language, out


def transcribe_gigaam(src, min_silence):
    """Russian: GigaAM v3 e2e-RNNT on Silero VAD speech spans, via onnx-asr (onnxruntime,
    no PyTorch). Verbatim: aborted takes and swears stay as their own segments instead of
    being merged into one clean sentence with stretched word spans like Whisper does (see
    references/lessons.md). Segments are the VAD spans; tokens carry start times only, so
    a word ends where the next one starts (or at the span end). No confidence ("p")."""
    import onnx_asr
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    wav = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    model = onnx_asr.load_model("gigaam-v3-e2e-rnnt").with_vad(
        onnx_asr.load_vad("silero"), min_silence_duration_ms=min_silence * 1000, speech_pad_ms=40).with_timestamps()
    out = []
    for seg in model.recognize(wav):
        words, new = [], True
        for tok, t in zip(seg.tokens, seg.timestamps):
            new = new or tok.startswith(" ")  # a bare " " token starts the next word
            if not tok.strip():
                continue
            if new or not words:
                words.append({"w": tok.strip(), "s": round(seg.start + t, 2)})
            else:
                words[-1]["w"] += tok
            new = tok.endswith(" ")
        for w, nxt in zip(words, words[1:] + [None]):
            w["e"] = round(nxt["s"] if nxt else seg.end, 2)
        out.append({"start": round(seg.start, 2), "end": round(seg.end, 2),
                    "text": " ".join(w["w"] for w in words), "words": words})
    return out


def cmd_transcribe(a):
    work, p = project(a.work)
    lang = detect_language(p["source"]) if a.lang == "auto" else a.lang
    engine = a.engine if a.engine != "auto" else ("gigaam" if lang == "ru" else "whisper")
    if engine == "gigaam":
        if lang != "ru":
            die(f"gigaam is Russian-only (language={lang}); use --engine whisper")
        out, model = transcribe_gigaam(p["source"], a.min_silence), "gigaam-v3-e2e-rnnt+silero"
    else:
        lang, out = transcribe_whisper(p["source"], lang, a.model, a.prompt)
        model = a.model
    save(work / "words.json", {"language": lang, "engine": engine, "model": model, "segments": out})
    lines = [f"[{ts(s['start'])}] {s['text']}" for s in out]
    (work / "transcript.txt").write_text("\n".join(lines) + "\n")
    nw = sum(len(s["words"]) for s in out)
    print(f"{work / 'transcript.txt'}: {len(out)} lines, {nw} words, language={lang}, engine={engine}")


def all_words(work):
    f = work / "words.json"
    if not f.exists():
        return []
    return [w for s in load(f)["segments"] for w in s["words"]]


def silences(src, noise, mind, start=None, dur=None):
    """Silence runs (source seconds) via ffmpeg silencedetect, optionally scoped to
    [start, start+dur) - the reported timestamps are offset back to absolute source
    time (silencedetect reports them relative to the seek point when -ss is used)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", src, "-af", f"silencedetect=noise={noise}dB:d={mind}", "-vn", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    off = start or 0.0
    out, s0 = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            s0 = float(m.group(1))
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and s0 is not None:
            out.append((s0 + off, float(m.group(1)) + off))
            s0 = None
    return out


# ---------- takes ----------

def norm_word(w):
    return re.sub(r"[^\wё]", "", w.lower())


def score_take(words, st, en):
    """finished/mumbled/stutter stats for the words falling inside [st, en)."""
    sw = [w for w in words if st <= (w["s"] + w["e"]) / 2 < en]
    norm = [norm_word(w["w"]) for w in sw]
    stutter = sorted({norm[j] for j in range(1, len(norm)) if norm[j] and norm[j] == norm[j - 1]})
    ps = [w["p"] for w in sw if "p" in w]  # gigaam words carry no confidence
    last = sw[-1]["w"] if sw else ""
    flags = []
    if not sw:
        flags.append("no_words")
    if ps and min(ps) < 0.6:
        flags.append("low_confidence")
    if stutter:
        flags.append("stutter")
    if any(PROFANITY.match(x) for x in norm):
        flags.append("profanity")
    if sw and not re.search(r"[.!?…,;:]['\")»]*$", last):
        flags.append("unfinished")
    return {
        "start": st, "end": en, "dur": round(en - st, 2),
        "text": " ".join(w["w"] for w in sw),
        "avg_p": round(sum(ps) / len(ps), 2) if ps else 0.0,
        "min_p": round(min(ps), 2) if ps else 0.0,
        "low_conf_words": [w["w"] for w in sw if w.get("p", 1) < 0.6],
        "stutter": stutter,
        "flags": flags,
    }


def opening(text, n=3):
    return [t for t in (norm_word(x) for x in text.split()) if t][:n]


def mark_retakes(takes, window):
    """Repeated attempts: when a later take (starting within `window` s) opens with the same
    first words as an earlier one, everything from the earlier take up to the later one is
    an abandoned attempt - flag it "retake" and point at the take that replaces it. Needs a
    verbatim transcript (gigaam); Whisper merges attempts into one segment and hides them."""
    for i, t in enumerate(takes):
        op = opening(t["text"])
        if len(op) < 2:
            continue
        for j in range(len(takes) - 1, i, -1):  # the last matching attempt wins
            if takes[j]["start"] - t["start"] <= window and opening(takes[j]["text"], len(op)) == op:
                for k in range(i, j):
                    if "retake" not in takes[k]["flags"]:
                        takes[k]["flags"].append("retake")
                        takes[k]["replaced_by"] = j
                break


def cmd_takes(a):
    work, p = project(a.work)
    words_doc = load(work / "words.json") if (work / "words.json").exists() else None
    if not words_doc:
        die("no words.json - run vid.py transcribe first")
    words = [w for s in words_doc["segments"] for w in s["words"]]
    if a.edl:
        segs = load(a.edl)["segments"]
    elif words_doc.get("engine") == "gigaam":
        segs = words_doc["segments"]  # already Silero VAD spans, verbatim: no sub-split needed
    else:
        # Default unit is Whisper's own sentence segments, not a silence-cut EDL:
        # dB-based silence cuts fragment one good sentence into pieces that each look
        # "unfinished" on their own. But also sub-split on any gap clearly too long to
        # be a mid-sentence breath (default 1s) - Whisper's VAD sometimes merges two
        # separate spoken attempts into one segment instead of breaking between them,
        # which otherwise hides a retake as a single implausibly-long word timestamp
        # (see .claude/skills/edit-video/references/lessons.md, Takes).
        segs = []
        for s in words_doc["segments"]:
            cur = s["start"]
            for gs, ge in silences(p["source"], -35, 0.03, s["start"], s["end"] - s["start"]):
                if ge - gs > a.retake_gap:
                    if gs - cur > 0.15:
                        segs.append({"start": cur, "end": gs})
                    cur = ge
            if s["end"] - cur > 0.15:
                segs.append({"start": cur, "end": s["end"]})
    out = [{"i": i, **score_take(words, s["start"], s["end"])} for i, s in enumerate(segs)]
    mark_retakes(out, a.retake_window)
    save(work / "takes.json", out, 1)
    nflag = sum(1 for t in out if t["flags"])
    print(f"{work / 'takes.json'}: {len(out)} takes, {nflag} flagged for review")


# ---------- mistakes (word-level cut candidates, review-gated) ----------

STOP = set("""и в во не что он на я с со как а то все всё она так его но да ты к у же вы за бы по только ее её мне
было вот от меня еще ещё нет о из ему когда даже ну ли если уже или ни быть был него до вас там потом себя ей может
они тут где есть надо ней для мы тебя их чем была сам без чего раз тоже себе под будет тогда кто этот того потому
этого какой здесь этом чтобы сейчас были можно при после над больше тот через эти нас про всего них это эту
a an and are as at be but by for from i if in is it of on or so that this to was we with you your they their there
what when where why how like um uh""".split())


def signature(text, n=18):
    toks = [norm_word(t) for t in text.split()]
    return [t[:5] for t in toks if len(t) > 2 and t not in STOP][:n]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def mistake_candidates(words, takes):
    """Cut candidates on the SOURCE timeline:
    stutters, retakes (take flags + content overlap), abandoned clauses at the end of an
    unfinished take, profanity. Each has a proposed cut range and a default recommendation;
    a reviewer decides (emphasis, listing and rhetorical doubling look like stutters)."""
    out = []
    ctx = lambda i: " ".join(w["w"] for w in words[max(0, i - 6):i + 6])
    for i in range(1, len(words)):
        a, b = words[i - 1], words[i]
        if norm_word(a["w"]) and norm_word(a["w"]) == norm_word(b["w"]) and b["s"] - a["e"] < 0.5:
            out.append({"type": "stutter", "start": a["s"], "end": b["s"], "recommend": "cut",
                        "removes": a["w"], "context": ctx(i)})
    for t in takes:
        if "retake" in t["flags"]:
            nxt = takes[t["replaced_by"]]
            out.append({"type": "retake", "take": t["i"], "start": t["start"], "end": t["end"], "recommend": "cut",
                        "removes": t["text"], "keeps": nxt["text"], "note": f"same opening as take {nxt['i']}"})
        if "profanity" in t["flags"]:
            out.append({"type": "profanity", "take": t["i"], "start": t["start"], "end": t["end"],
                        "recommend": "cut", "removes": t["text"]})
    sigs = [signature(t["text"]) for t in takes]
    for i, t in enumerate(takes):
        if "retake" in t["flags"] or len(sigs[i]) < 2:
            continue
        for j in range(i + 1, min(len(takes), i + 9)):
            if takes[j]["start"] - t["end"] > 75:
                break
            sc = jaccard(sigs[i], sigs[j])
            if sc >= 0.34:
                out.append({"type": "retake", "take": t["i"], "start": t["start"], "end": t["end"],
                            "recommend": "cut" if sc >= 0.45 else "review", "removes": t["text"],
                            "keeps": takes[j]["text"], "note": f"content overlap {sc:.2f} with take {takes[j]['i']}"})
                break
    for i, t in enumerate(takes[:-1]):
        if "unfinished" not in t["flags"] or CUT_FLAGS & set(t["flags"]):
            continue
        tw = [w for w in words if t["start"] <= (w["s"] + w["e"]) / 2 < t["end"]]
        k = max((n + 1 for n, w in enumerate(tw[:-1]) if re.search(r"[.,!?;:…]['\")»]*$", w["w"])), default=0)
        tail, nxt = tw[k:], takes[i + 1]
        if not tail or len(tail) > 6:
            continue
        restart = opening(nxt["text"], 1) == opening(" ".join(w["w"] for w in tail), 1)
        out.append({"type": "false_start", "take": t["i"], "start": tail[0]["s"], "end": t["end"],
                    "recommend": "cut" if restart else "review",
                    "removes": " ".join(w["w"] for w in tail), "keeps": nxt["text"],
                    "context": t["text"], "note": "abandoned clause at the end of an unfinished take"
                    + (", next take restarts with the same word" if restart else "")})
    out.sort(key=lambda c: c["start"])
    for n, c in enumerate(out):
        c["id"] = f"m{n + 1}"
        c["start"], c["end"] = round(c["start"], 3), round(c["end"], 3)
    return out


def cmd_mistakes(a):
    work, p = project(a.work)
    if not (work / "takes.json").exists():
        die("no takes.json - run vid takes first")
    cands = mistake_candidates(all_words(work), load(work / "takes.json"))
    save(work / "mistakes.json", cands, 1)
    for c in cands:
        print(f"{c['id']:>4} {c['type']:<11} {c['recommend']:<6} {ts(c['start'])}-{ts(c['end'])}  "
              f"cut: {c['removes'][:70]!r}" + (f"  ({c['note']})" if c.get("note") else ""))
    print(f"{work / 'mistakes.json'}: {len(cands)} candidates; approve into {work / 'cuts.json'}")


# ---------- cut ----------

CUT_FLAGS = {"retake", "profanity", "no_words"}


def cmd_cut(a):
    """Kept takes -> out/cut.mp4 (one clip, source frame, constant fps, keyframe every
    second so HyperFrames seeks don't freeze) + cut.json (segments and words on the
    cut timeline). Cutting stays here; the composition only plays the clip."""
    work, p = project(a.work)
    if not (work / "takes.json").exists():
        die("no takes.json - run vid takes first")
    takes = load(work / "takes.json")
    cpath = Path(a.cuts) if a.cuts else work / "cuts.json"
    cuts = load(cpath) if cpath.exists() else {}
    keep = {int(x) for x in a.keep.split(",")} if a.keep else set()
    drop = {int(x) for x in a.drop.split(",")} if a.drop else set()
    keep |= set(cuts.get("keep", []))
    drop |= set(cuts.get("drop", []))
    fps = round(p["fps"])
    snap = lambda t: round(t * fps) / fps  # frame-exact edges: video and audio of a segment stay equal
    spans = [(t["start"], t["end"], t["i"]) for t in takes
             if t["i"] in keep or (t["i"] not in drop and not CUT_FLAGS & set(t["flags"]))]
    for c in sorted(cuts.get("cuts", []), key=lambda c: c["start"]):  # word-level cuts inside kept takes
        nxt = []
        for st, en, ti in spans:
            if c["end"] <= st or c["start"] >= en:
                nxt.append((st, en, ti))
                continue
            if c["start"] - st >= 0.2:
                nxt.append((st, c["start"], ti))
            if en - c["end"] >= 0.2:
                nxt.append((c["end"], en, ti))
        spans = nxt
    segs = [{"start": round(snap(st), 3), "end": round(snap(en), 3), "take": ti} for st, en, ti in spans]
    segs = [s for s in segs if s["end"] > s["start"]]
    if not segs:
        die("no takes left to keep")
    offs, total = seg_times(segs)
    for s, o in zip(segs, offs):
        s["out"] = round(o, 3)
    words = []
    for w in all_words(work):
        # the kept segment the word overlaps most; its span is clipped to that segment
        ov = [(min(w["e"], s["end"]) - max(w["s"], s["start"]), s, o) for s, o in zip(segs, offs)]
        d, s, o = max(ov, key=lambda x: x[0])
        if d <= 0 or (d < 0.1 and d < 0.5 * (w["e"] - w["s"])):  # a sliver left by a cut edge: not a spoken word
            continue
        words.append({"w": w["w"], "s": round(o + max(w["s"], s["start"]) - s["start"], 3),
                      "e": round(o + min(w["e"], s["end"]) - s["start"], 3)})
    out = work / "out" / "cut.mp4"
    cmd, parts = ["ffmpeg", "-v", "error", "-y"], []
    for s in segs:
        cmd += ["-ss", f"{s['start']:.3f}", "-t", f"{s['end'] - s['start'] + 0.1:.3f}", "-i", p["source"]]
    if not p["audio"]:
        cmd += ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    trc = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_transfer",
               "-of", "csv=p=0", p["source"]]).strip().strip(",")
    # iPhone HLG/PQ -> SDR bt709: HyperFrames otherwise promotes the whole render to HDR
    tonemap = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
               "zscale=t=bt709:m=bt709:r=tv,format=yuv420p," if trc in ("arib-std-b67", "smpte2084") else "")
    for i, s in enumerate(segs):
        d = s["end"] - s["start"]
        n = round(d * fps)
        fade = min(0.01, d / 4)  # no clicks at the joins
        parts.append(f"[{i}:v:0]{tonemap}fps={fps},setsar=1,tpad=stop_mode=clone:stop=2,trim=end_frame={n},setpts=PTS-STARTPTS[v{i}]")
        if p["audio"]:
            parts.append(f"[{i}:a:0]aresample=48000,apad,atrim=end_sample={round(n / fps * 48000)},asetpts=PTS-STARTPTS,"
                         f"afade=t=in:d={fade},afade=t=out:st={d - fade:.3f}:d={fade}[a{i}]")
    if p["audio"]:
        parts.append("".join(f"[v{i}][a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[v][a]")
        amap = "[a]"
    else:
        parts.append("".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]")
        amap = f"{len(segs)}:a:0"
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", amap,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(out)]
    run(cmd)
    real = probe(out)["duration"]
    lines = []  # one line per kept take on the cut timeline, for planning the composition
    for sg in segs:
        e = sg["out"] + sg["end"] - sg["start"]
        text = " ".join(w["w"] for w in words if sg["out"] - 1e-6 <= w["s"] < e)
        lines.append(f"[{sg['out']:.2f}-{e:.2f}] {text}")
    (work / "cut.txt").write_text("\n".join(lines) + "\n")
    save(work / "cut.json", {"source": p["source"], "video": str(out), "fps": fps,
                             "width": p["width"], "height": p["height"], "duration": round(real, 3),
                             "segments": segs, "words": words}, 1)
    print(f"{out}  {len(segs)} takes, {ts(real)} (planned {total:.2f}s), {len(words)} words"
          f"{', HDR tone-mapped to SDR' if tonemap else ''} -> {work / 'cut.json'}")
    if abs(real - total) > 0.1:
        print(f"warning: clip is {real - total:+.2f}s off the planned length; word timings may drift", file=sys.stderr)


def match_words(words, phrase, start=0.0, end=1e9):
    """Best contiguous run of `words` (source-time order) matching phrase's tokens.
    None if the best run scores under 0.5 (fraction of tokens matching by position)."""
    ph = [norm_word(t) for t in phrase.split() if norm_word(t)]
    if not ph:
        return None
    cand = [w for w in words if start <= w["s"] <= end]
    norm = [norm_word(w["w"]) for w in cand]
    n = len(ph)
    best = None
    for i in range(0, max(0, len(cand) - n + 1)):
        score = sum(1 for x, y in zip(norm[i:i + n], ph) if x == y) / n
        if best is None or score > best[0]:
            best = (score, i)
    if not best or best[0] < 0.5:
        return None
    score, i = best
    ws = cand[i:i + n]
    return {"start": ws[0]["s"], "end": ws[-1]["e"], "text": " ".join(w["w"] for w in ws), "score": round(score, 2)}


def cmd_say(a):
    work, p = project(a.work)
    words = load(work / "cut.json")["words"] if a.cut else all_words(work)
    if not words:
        die("no words.json - run vid.py transcribe first")
    m = match_words(words, a.text, a.start, a.end)
    if not m:
        die(f"no good match for '{a.text}' (nothing scored >= 0.5)")
    print(json.dumps(m, ensure_ascii=False))


def frames(src, fps, width):
    """Yield (t, bgr frame) sampled at fps, scaled to width, via one ffmpeg pipe."""
    import numpy as np
    info = probe(src)
    h = int(round(info["height"] * width / info["width"] / 2) * 2)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={fps},scale={width}:{h}",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size, i = width * h * 3, 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, width, 3)
        i += 1
    proc.wait()


def median_box(boxes):
    import numpy as np
    return [int(v) for v in np.median(np.array([b[:4] for b in boxes]), axis=0)] if boxes else None


# ---------- timeline helpers ----------

def seg_times(segs):
    out, t = [], 0.0
    for s in segs:
        out.append(t)
        t += s["end"] - s["start"]
    return out, t


def ass_time(t):
    h, r = divmod(max(0, t), 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def srt(ww, total):
    out, n, cur = [], 1, []
    def flush():
        nonlocal n
        if cur:
            f = lambda t: ass_time(t).replace(".", ",").rjust(11, "0") + "0"
            out.append(f"{n}\n{f(cur[0]['s'])} --> {f(min(total, cur[-1]['e'] + 0.2))}\n{' '.join(x['w'] for x in cur)}\n")
            n += 1
    for w in ww:
        if cur and (len(cur) >= 10 or w["s"] - cur[-1]["e"] > 0.8 or re.search(r"[.?!]$", cur[-1]["w"])):
            flush()
            cur = []
        cur.append(w)
    flush()
    return "\n".join(out)


def cmd_spans(a):
    """Suggested face / visual parts for template "alternate" (3-7 s each, cut in word gaps),
    with the words spoken in each; paste the visual ones into composition.json "visual"."""
    work, p = project(a.work)
    cut = load(work / "cut.json")
    parts = alternate_spans(cut["words"], cut["duration"], a.min, a.max)
    for x in parts:
        text = " ".join(w["w"] for w in cut["words"] if x["start"] <= (w["s"] + w["e"]) / 2 < x["end"])
        print(f"{x['kind']:6} {x['start']:6.2f}-{x['end']:6.2f} ({x['end'] - x['start']:.1f} s)  {text}")
    print(json.dumps({"visual": [{"start": x["start"], "end": x["end"]} for x in parts if x["kind"] == "visual"]}))


# ---------- compose (HyperFrames owns the frame) ----------

STAGE_KINDS = {"title", "number", "list", "chat", "compare", "window", "custom"}
LUCIDE = HYPERFRAMES / "node_modules" / "lucide-static"
# composition.json "template": which HyperFrames frame renders the clip (references/templates.md)
TEMPLATES = {"visuals": "talking-head.html", "simple": "simple.html", "alternate": "talking-head.html"}
ALT_SPAN = (3.0, 7.0)  # template "alternate": every face / visual part lasts 3-7 s
# tokens every template uses; brand/theme.css must define all of them
THEME_TOKENS = ["--bg", "--panel", "--panel-2", "--text", "--muted", "--overlay-text", "--accent", "--accent-hi",
                "--edge", "--line", "--vignette", "--font-sans", "--font-accent", "--accent-style"]


def theme_css(warn=None):
    """brand/theme.css, with brand/fonts copied to hyperframes/assets/brand/fonts (its url()s
    point there). Missing tokens or font files are warnings: the render still runs."""
    f = BRAND / "theme.css"
    if not f.exists():
        die(f"no {f}: restore it from git or run the style skill")
    css = f.read_text()
    fonts = HYPERFRAMES / "assets" / "brand" / "fonts"
    shutil.rmtree(fonts, ignore_errors=True)
    shutil.copytree(BRAND / "fonts", fonts) if (BRAND / "fonts").is_dir() else fonts.mkdir(parents=True)
    if warn is not None:
        body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        warn += [f"brand/theme.css: token {t} is missing" for t in THEME_TOKENS if not re.search(rf"{t}\s*:", body)]
        warn += [f"brand/theme.css: font file {u} not found in brand/fonts"
                 for u in re.findall(r"url\([\"']?assets/brand/fonts/([^\"')]+)", body) if not (fonts / u).exists()]
    return css


def template_html(name, warn=None):
    """A composition from hyperframes/compositions with the brand theme injected."""
    return (HYPERFRAMES / "compositions" / name).read_text().replace("/*__THEME__*/", theme_css(warn))


def icon_svg(name):
    f = LUCIDE / "icons" / f"{name}.svg"
    if not f.exists():
        return None
    svg = re.sub(r"<!--.*?-->", "", f.read_text(), flags=re.S)
    # size/class only off the outer <svg> tag: inner <rect> shapes need their width/height
    return re.sub(r"<svg\b[^>]*>", lambda m: re.sub(r'\s(class|width|height)="[^"]*"', "", m.group()), svg,
                  count=1).strip()


def cmd_icons(a):
    """Lucide icon names matching a word (name or tag), for beats' icon / <i data-icon>."""
    tags = json.loads((LUCIDE / "tags.json").read_text())
    q = a.query.lower()
    hits = [n for n, t in tags.items() if q in n or any(q in x.lower() for x in t)]
    print(" ".join(sorted(hits, key=lambda n: (q not in n, len(n)))[:a.n]) or f"no icon for '{a.query}'")


def prepare_beat(b, i, warn):
    """Inline icons, sanitise custom HTML/CSS (no scripts, handlers or remote URLs)."""
    b = dict(b)
    if b.get("icon"):
        b["icon_svg"] = icon_svg(b["icon"]) or ""
        if not b["icon_svg"]:
            warn.append(f"beat {i}: no icon '{b['icon']}' (vid icons WORD lists names)")
    if b.get("kind") == "custom":
        html = b.get("html", "")
        html = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)
        html = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", html, flags=re.I)
        html = re.sub(r"""\s(src|href)\s*=\s*["'](https?:)?//[^"']*["']""", "", html, flags=re.I)

        def icon(m):
            svg = icon_svg(m.group(1))
            if not svg:
                warn.append(f"beat {i}: no icon '{m.group(1)}'")
            return f'<i class="icon {m.group(2) or ""}">{svg or ""}</i>'
        b["html"] = re.sub(r"""<i\s+data-icon=["']([\w-]+)["'](?:\s+class=["']([^"']*)["'])?\s*/?>(?:\s*</i>)?""", icon, html)
        if "data-icon" in b["html"]:
            warn.append(f"beat {i}: an <i data-icon> tag was not understood; write <i data-icon=\"NAME\" class=\"...\"></i>")
        b["css"] = re.sub(r"url\((['\"]?)(https?:)?//[^)]*\)", "none", b.get("css", ""))
        for n, an in enumerate(b.get("anim", [])):
            if an.get("at") is not None and not b["start"] - 0.05 <= an["at"] <= b["end"]:
                warn.append(f"beat {i} anim {n}: at {an['at']} outside the beat {b['start']}-{b['end']}")
    return b


def face_box(video, every=1.0):
    """Median of the largest face per sample (YuNet), in video pixels; None if no face."""
    import cv2
    info = probe(video)
    sw = min(640, info["width"])
    k = info["width"] / sw
    det, boxes = None, []
    for _, img in frames(str(video), 1 / every, sw):
        if det is None:
            det = cv2.FaceDetectorYN.create(str(YUNET), "", (img.shape[1], img.shape[0]), 0.7)
        _, f = det.detect(img)
        if f is not None and len(f):
            r = max(f, key=lambda r: r[2] * r[3])
            boxes.append([v * k for v in r[:4]])
    b = median_box(boxes)
    return {"x": b[0], "y": b[1], "w": b[2], "h": b[3]} if b else None


def alternate_spans(words, dur, lo=ALT_SPAN[0], hi=ALT_SPAN[1], target=5.0):
    """Default parts for template "alternate": face, visual, face, ... each lo-hi s, cut in the
    widest word gap near `target` s, preferring sentence ends, then commas.
    Returns [{"start", "end", "kind": "face"|"visual"}]."""
    stop = lambda w: 0.6 if re.search(r"[.!?…]$", w) else 0.2 if re.search(r"[,;:—-]$", w) else 0.0
    gaps = [((x["e"] + y["s"]) / 2, y["s"] - x["e"] + stop(x["w"])) for x, y in zip(words, words[1:])]
    edges, cur = [0.0], 0.0
    while dur - cur > hi:
        c = [(g - 0.1 * abs(t - cur - target), t) for t, g in gaps if cur + lo <= t <= cur + hi and dur - t >= lo]
        cur = round(max(c)[1] if c else cur + target, 3)
        edges.append(cur)
    if dur - cur < lo and len(edges) > 1 and dur - edges[-2] <= hi:
        edges.pop()  # a short tail joins the previous part
    edges.append(round(dur, 3))
    return [{"start": a, "end": b, "kind": ("face", "visual")[i % 2]}
            for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def check_alternate(comp, dur):
    """Template "alternate": visual spans and the face parts between them last 3-7 s, face first."""
    warn, vis = [], sorted(comp["visual"], key=lambda v: v["start"])
    lo, hi = ALT_SPAN
    if not vis:
        warn.append("no visual spans: template alternate needs face / visual parts (vid spans WORK suggests them)")
    edges = [0.0] + [t for v in vis for t in (v["start"], v["end"])] + [dur]
    for i, (x, y) in enumerate(zip(edges, edges[1:])):
        kind = ("face", "visual")[i % 2]
        if y - x < -0.01:
            warn.append(f"visual spans overlap or run past the end near {x:.2f} s")
        elif kind == "face" and y - x < 0.05 and (i == 0 or y >= dur - 0.05):
            if i == 0:
                warn.append("starts with visuals: the face part comes first (the hook)")
        elif not lo - 0.05 <= y - x <= hi + 0.05:
            warn.append(f"{kind} part {x:.2f}-{y:.2f} s lasts {y - x:.1f} s, keep parts {lo:.0f}-{hi:.0f} s")
    return warn


def check_composition(comp, dur, spans=None, margin=0.4, where="a pip span (stage shows only while the speaker "
                                                                 "is a circle)"):
    """Warnings for composition.json against the clip; empty list = fine. Beats must sit inside
    `spans` (default: the pip spans) at least `margin` s from both ends."""
    warn, pips = [], comp["pip"] if spans is None else spans
    for i, b in enumerate(comp["beats"]):
        tag = f"beat {i} ({b.get('kind')} {b.get('start')}-{b.get('end')})"
        if b.get("kind") not in STAGE_KINDS:
            warn.append(f"{tag}: unknown kind, skipped; use one of {sorted(STAGE_KINDS)}")
            continue
        if not 0 <= b["start"] < b["end"] <= dur + 0.05:
            warn.append(f"{tag}: outside 0-{dur:.2f} or empty")
        elif b["end"] - b["start"] < 0.8:
            warn.append(f"{tag}: shorter than 0.8 s, unreadable")
        if not any(p["start"] + margin - 0.01 <= b["start"] and b["end"] <= p["end"] - margin + 0.01 for p in pips):
            warn.append(f"{tag}: not inside {where}")
    bs = sorted((b for b in comp["beats"] if b.get("kind") in STAGE_KINDS), key=lambda b: b["start"])
    for x, y in zip(bs, bs[1:]):
        if y["start"] < x["end"] - 0.01:
            warn.append(f"beats at {x['start']} and {y['start']} overlap")
    for x, y, z in zip(bs, bs[1:], bs[2:]):
        if x["kind"] == y["kind"] == z["kind"] != "custom":
            warn.append(f"three {x['kind']} beats in a row from {x['start']}: vary the mechanism")
    stock = sum(b["kind"] != "custom" for b in bs)
    if bs and stock / len(bs) > 0.5:
        warn.append(f"{stock}/{len(bs)} beats are stock cards: design custom graphics from this video's content "
                    "(references/edit-rules.md §2)")
    for p in pips:  # stage gaps: >2.5 s with nothing new on the stage needs a reason
        edges = [p["start"] + margin] + [t for b in bs if p["start"] <= b["start"] < p["end"] for t in (b["start"], b["end"])] + [p["end"] - margin]
        for x, y in zip(edges[::2], edges[1::2]):
            if y - x > 2.5:
                warn.append(f"stage empty {x:.1f}-{y:.1f} s")
    return warn


def fix_words(words, fixes):
    """Caption spelling fixes {"recognised words": "right text"}: every run of words whose
    normalised form matches the key becomes one word (the fix + the last word's punctuation)."""
    out, keys = [], [([norm_word(x) for x in k.split()], v) for k, v in fixes.items()]
    i = 0
    while i < len(words):
        for k, v in keys:
            n = len(k)
            if k and [norm_word(w["w"]) for w in words[i:i + n]] == k:
                tail = re.search(r"[^\w]*$", words[i + n - 1]["w"]).group()
                out.append({"w": v + tail, "s": words[i]["s"], "e": words[i + n - 1]["e"]})
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


def cmd_compose(a):
    """cut.json + composition.json -> out/final.mp4 (+ .srt) rendered by HyperFrames.
    Without composition.json: template "visuals" (talking-head.html), speaker full -> circle
    -> full, captions, empty stage. composition.json {"template": "simple"} (simple.html):
    speaker full screen throughout, captions only, no pip/beats. {"template": "alternate"}
    (talking-head.html, layout alternate): full-screen face parts and full-frame graphics parts
    ("visual" spans, voice continues) take turns every 3-7 s, captions throughout."""
    work, p = project(a.work)
    if not (work / "cut.json").exists():
        die("no cut.json - run vid cut first")
    cut = load(work / "cut.json")
    dur, fps = cut["duration"], cut["fps"]
    cpath = Path(a.composition) if a.composition else work / "composition.json"
    comp = load(cpath) if cpath.exists() else {}
    tname = comp.get("template", "visuals")
    if tname not in TEMPLATES:
        die(f"template {tname!r}: use one of {sorted(TEMPLATES)}")
    warn, face = [], None
    if tname == "visuals":
        comp.setdefault("pip", [{"start": 1.2, "end": round(dur - 1.0, 3)}] if dur > 5 else [])
        comp.setdefault("beats", [])
        warn = check_composition(comp, dur)
        face = face_box(cut["video"])
        if not face:
            warn.append("no face found in the clip: the circle shows the frame centre")
    elif tname == "alternate":
        comp.setdefault("visual", [{"start": x["start"], "end": x["end"]}
                                   for x in alternate_spans(cut["words"], dur) if x["kind"] == "visual"])
        comp.setdefault("beats", [])
        comp["pip"] = []
        warn = check_alternate(comp, dur) + check_composition(comp, dur, comp["visual"], 0.0, "a visual span")
        warn += [f"visual span {v['start']}-{v['end']} has no beats" for v in comp["visual"]
                 if not any(v["start"] - 0.01 <= b.get("start", -1) < v["end"] for b in comp["beats"])]
    elif comp.get("pip") or comp.get("beats"):
        warn.append("template simple ignores pip/beats; drop them or switch template to visuals")
    words = fix_words(cut["words"], comp.get("fix", {}))
    variables = {
        "src": {"w": cut["width"], "h": cut["height"]}, "face": face, "pip": comp.get("pip", []),
        "words": words, "caption": comp.get("caption", True),
        "emph": {norm_word(k): v for k, v in comp.get("emph", {}).items()},
        "beats": [prepare_beat(b, i, warn) for i, b in enumerate(comp.get("beats", []))],
        "layout": "alternate" if tname == "alternate" else "pip", "visual": comp.get("visual", []),
    }
    out = Path(a.out) if a.out else work / "out" / "final.mp4"
    tmpdir = HYPERFRAMES / ".tmp"  # HyperFrames refuses an entry file outside its project dir
    tmpdir.mkdir(exist_ok=True)
    key = hashlib.sha1(str(work.resolve()).encode()).hexdigest()[:10]
    clip, html, vfile = tmpdir / f"{key}.mp4", tmpdir / f"{key}.html", tmpdir / f"{key}.json"
    clip.unlink(missing_ok=True)
    try:
        os.link(cut["video"], clip)
    except OSError:
        shutil.copy2(cut["video"], clip)
    template = template_html(TEMPLATES[tname], warn)
    frames_n = round(dur * fps)  # duration rounded down on the frame grid: 17.667 s would render 531 frames
    html.write_text(template.replace("__DURATION__", f"{math.floor(frames_n / fps * 1e4) / 1e4:.4f}")
                    .replace("__FPS__", str(fps)).replace("__VIDEO__", f".tmp/{clip.name}"))
    save(vfile, variables)
    try:
        run(["npx", "hyperframes", "render", "-c", f".tmp/{html.name}", "-o", str(out.resolve()),
             "--variables-file", str(vfile), "--fps", str(fps), "--quiet"], cwd=HYPERFRAMES)
    finally:
        if not a.keep_tmp:
            for f in (clip, html, vfile):
                f.unlink(missing_ok=True)
    out.with_suffix(".srt").write_text(srt(words, dur))
    real = probe(out)["duration"]
    detail = {"visuals": f", {len(comp.get('beats', []))} beats, pip {comp.get('pip', [])}",
              "alternate": f", {len(comp.get('beats', []))} beats, visual {comp.get('visual', [])}"}.get(tname, "")
    print(f"{out}  {ts(real)}, template {tname}{detail}")
    for w in warn:
        print(f"warning: {w}", file=sys.stderr)
    return {"out": str(out.resolve()), "duration": round(real, 2), "warnings": warn}


# ---------- carousel (static slides, same house style) ----------

CAROUSEL_RATIOS = {"4:5": (1080, 1350), "3:4": (1080, 1440), "1:1": (1080, 1080)}


def hf_json(cmd):
    """Run a HyperFrames command with --json; parsed stdout, or {} when it printed none."""
    r = subprocess.run(cmd + ["--json"], capture_output=True, text=True, cwd=HYPERFRAMES)
    try:
        return json.loads(r.stdout[r.stdout.index("{"):])
    except ValueError:
        return {}


def cmd_carousel(a):
    """WORK/slides.json -> WORK/out/NN.png (one per slide) + sheet.jpg + report.json, rendered
    by HyperFrames (compositions/carousel.html) and audited by `hyperframes check`."""
    work = Path(a.work)
    spec = load(work / "slides.json")
    slides, warn = spec.get("slides", []), []
    if not 2 <= len(slides) <= 20:
        die(f"{len(slides)} slides: Instagram takes 2-20")
    ratio = spec.get("ratio", "4:5")
    if ratio not in CAROUSEL_RATIOS:
        die(f"ratio {ratio}: use one of {sorted(CAROUSEL_RATIOS)}")
    w, h = CAROUSEL_RATIOS[ratio]
    brand = load(BRAND / "brand.json") if (BRAND / "brand.json").exists() else {}
    data = {"handle": spec.get("handle", brand.get("handle", "")), "next": spec.get("next", ""), "slides": []}
    for i, s in enumerate(slides):
        b = prepare_beat({"kind": "custom", "start": 0, "end": 1, "html": s.get("html", ""),
                          "css": s.get("css", "")}, i + 1, warn)
        data["slides"].append({"html": b["html"], "css": b["css"], "chrome": s.get("chrome", True)})
    warn = [x.replace("beat ", "slide ") for x in warn]
    stage = HYPERFRAMES / ".tmp" / ("car-" + hashlib.sha1(str(work.resolve()).encode()).hexdigest()[:10])
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    (stage / "assets").symlink_to(HYPERFRAMES / "assets")
    shutil.copy2(HYPERFRAMES / "hyperframes.json", stage)
    n = len(slides)
    (stage / "index.html").write_text(template_html("carousel.html", warn)
                                      .replace("__W__", str(w)).replace("__H__", str(h))
                                      .replace("__DURATION__", str(n))
                                      .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    try:
        rep = hf_json(["npx", "hyperframes", "check", str(stage)])
        for sec in ("runtime", "layout", "contrast"):
            for f in (rep.get(sec) or {}).get("findings", []):
                if f.get("severity") == "info":
                    continue
                t = f.get("time")
                where = f"slide {min(n, int(t) + 1)}" if isinstance(t, (int, float)) else sec
                warn.append(f"{where}: {f.get('code')}: {f.get('message', '')[:160]} {f.get('selector') or ''}".rstrip())
        if not rep:
            warn.append("hyperframes check printed no report")
        snaps = stage / "snaps"
        run(["npx", "hyperframes", "snapshot", str(stage), "--at", ",".join(f"{i + 0.5}" for i in range(n)),
             "--no-end", "--describe", "false", "-o", str(snaps)], cwd=HYPERFRAMES)
        out = work / "out"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        pngs = sorted(snaps.glob("frame-*.png"))
        if len(pngs) != n:
            die(f"snapshot wrote {len(pngs)} frames for {n} slides")
        for i, p in enumerate(pngs):
            shutil.move(p, out / f"{i + 1:02d}.png")
        tile_sheet(sorted(out.glob("*.png")), out / "sheet.jpg")
    finally:
        if not a.keep_tmp:
            shutil.rmtree(stage, ignore_errors=True)
    warn = list(dict.fromkeys(warn))
    save(work / "out" / "report.json", {"slides": n, "ratio": ratio, "size": [w, h], "warnings": warn}, 1)
    print(f"{work / 'out'}: {n} slides {w}x{h} ({ratio}), sheet {work / 'out' / 'sheet.jpg'}")
    for x in warn:
        print(f"warning: {x}")
    print("DONE" if not warn else f"{len(warn)} warnings: fix slides.json and rerun")


# ---------- edit: the whole flow, resumable ----------

def newer(out, *ins):
    """out exists and is newer than every existing input."""
    out = Path(out)
    return out.exists() and all(out.stat().st_mtime >= Path(i).stat().st_mtime for i in ins if Path(i).exists())


EDIT_STEPS = ["init", "transcribe", "takes", "mistakes", "cut", "compose", "sheet"]


def cmd_edit(a):
    """One call per video, rerun until done: every step whose output is older than its inputs
    runs; it stops at the two review points and says which file to write:
    cuts.json (approved cuts, from mistakes.json + transcript.txt) and composition.json
    (stage graphics, from cut.txt). --from STEP redoes STEP and everything after it."""
    src = local_path(a.video)
    work = Path(a.work) if a.work else default_work(src)
    force = EDIT_STEPS.index(a.from_step) if a.from_step else len(EDIT_STEPS)
    NS = argparse.Namespace
    step = lambda name: EDIT_STEPS.index(name) >= force
    words = work / "words.json"
    if words.exists() and "engine" not in load(words) and not step("transcribe"):
        die(f"{work} holds a legacy (pre-v2) transcript; use -w NEW_DIR or --from transcribe")
    if step("init") or not (work / "project.json").exists():
        cmd_init(NS(video=str(src), work=str(work)))
    if step("transcribe") or not newer(words, work / "project.json"):
        cmd_transcribe(NS(work=str(work), engine="auto", min_silence=0.15, model="large-v3-turbo",
                          lang=a.lang, prompt=None))
    lang = load(words).get("language", "?")
    if step("takes") or not newer(work / "takes.json", words):
        cmd_takes(NS(work=str(work), edl=None, retake_window=60.0, retake_gap=1.0))
    if step("mistakes") or not newer(work / "mistakes.json", work / "takes.json"):
        cmd_mistakes(NS(work=str(work)))
    print(f"language {lang}")
    if not (work / "cuts.json").exists():
        print(f"NEXT: review {work / 'mistakes.json'} against {work / 'transcript.txt'} and write "
              f"{work / 'cuts.json'} (see references/edit-rules.md §1), then rerun vid edit")
        return
    if step("cut") or not newer(work / "cut.json", work / "takes.json", work / "cuts.json"):
        cmd_cut(NS(work=str(work), keep=None, drop=None, cuts=None))
    if not (work / "composition.json").exists():
        print(f"NEXT: design {work / 'composition.json'} from {work / 'cut.txt'} (see references/edit-rules.md §2; "
              "default template \"visuals\" designs stage graphics, \"template\": \"simple\" is full-screen "
              "captions only, no graphics, \"template\": \"alternate\" swaps face and full-frame graphics "
              "parts every 3-7 s, vid spans WORK suggests them), then rerun vid edit")
        return
    final = work / "out" / "final.mp4"
    tname = load(work / "composition.json").get("template", "visuals")
    template = HYPERFRAMES / "compositions" / TEMPLATES.get(tname, "talking-head.html")
    rep_path = work / "out" / "report.json"
    if step("compose") or not newer(final, work / "cut.json", work / "composition.json", template,
                                    BRAND / "theme.css"):
        rep = cmd_compose(NS(work=str(work), composition=None, out=None, keep_tmp=False))
        save(rep_path, rep, 1)
    sheet = work / "out" / "final.sheet.jpg"
    if step("sheet") or not newer(sheet, final):
        cmd_sheet(NS(video=str(final), n=16, cols=8, width=200, out=str(sheet)))
    rep = load(rep_path) if rep_path.exists() else {}
    print(f"DONE: {final} ({rep.get('duration', '?')} s), sheet {sheet}, "
          f"{len(rep.get('warnings', []))} warnings" + "".join(f"\n  warning: {w}" for w in rep.get("warnings", [])))


# ---------- contact sheet ----------

def tile_sheet(images, out, cols=5, width=216):
    """Tile image files (same aspect) into one jpg, downscaled to `width` px each."""
    import cv2
    import numpy as np
    tiles = [cv2.imread(str(f)) for f in images]
    tiles = [cv2.resize(t, (width, round(t.shape[0] * width / t.shape[1]))) for t in tiles if t is not None]
    if not tiles:
        return
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def cmd_sheet(a):
    import cv2
    import numpy as np
    info = probe(a.video)
    n, cols = a.n, a.cols
    times = [info["duration"] * (i + 0.5) / n for i in range(n)]
    tiles = []
    for t in times:
        buf = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", a.video, "-frames:v", "1",
                              "-vf", f"scale={a.width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR) if buf else None
        if img is None:
            continue
        cv2.putText(img, ts(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, ts(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        die("no frames extracted")
    h, w = tiles[0].shape[:2]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack([cv2.resize(x, (w, h)) for x in tiles[i:i + cols]]) for i in range(0, len(tiles), cols)]
    out = a.out or str(Path(a.video).with_suffix(".sheet.jpg"))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(out)


# ---------- preview: the brand style on sample frames ----------

PREVIEW_TEXT = "Так выглядят субтитры в вашем стиле: цвета, шрифты и акцентные слова"


def cmd_preview(a):
    """brand/preview/: carousel.jpg (examples/carousel rendered) and video.jpg (frames of a
    synthetic clip in both video templates), so a style change can be checked in one look."""
    NS = argparse.Namespace
    out = BRAND / "preview"
    out.mkdir(exist_ok=True)
    if not a.video_only:
        car = out / "carousel"
        shutil.rmtree(car, ignore_errors=True)
        car.mkdir()
        shutil.copy2(ROOT / "examples" / "carousel" / "slides.json", car / "slides.json")
        cmd_carousel(NS(work=str(car), keep_tmp=False))
        shutil.copy2(car / "out" / "sheet.jpg", out / "carousel.jpg")
    if a.carousel_only:
        return
    work = out / "video"
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)
    clip, dur, fps = work / "out" / "cut.mp4", 7.0, 30
    # a grey head-and-shoulders silhouette on a gradient stands in for the speaker
    body = ("if(lt(hypot(X-540,(Y-720)*0.8),190),170,if(gt(Y,1000)*lt(hypot((X-540)*0.8,Y-1640),380),"
            "120,30+40*Y/H))")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=black:s=1080x1920:d={dur}:r={fps},format=yuv420p,geq=lum='{body}':cb=132:cr=124",
         "-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=48000:cl=stereo",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(fps), "-c:a", "aac", "-shortest", str(clip)])
    toks = PREVIEW_TEXT.split()
    step = (dur - 1.0) / len(toks)
    words = [{"w": w, "s": round(0.5 + i * step, 3), "e": round(0.5 + (i + 1) * step - 0.05, 3)} for i, w in enumerate(toks)]
    save(work / "project.json", {"source": str(clip), **probe(clip)}, 2)
    save(work / "cut.json", {"source": str(clip), "video": str(clip), "fps": fps, "width": 1080, "height": 1920,
                             "duration": dur, "segments": [], "words": words}, 1)
    emph = {"стиле": "bold", "акцентные": "serif"}
    comps = {
        "visuals": {"template": "visuals", "pip": [{"start": 0.8, "end": 6.4}], "emph": emph, "beats": [
            {"kind": "title", "start": 1.3, "end": 3.5, "text": "Ваш *фирменный* стиль", "sub": "заголовок, акцент, подпись"},
            {"kind": "custom", "start": 3.6, "end": 6.0,
             "html": '<div class="row"><div class="card hot"><div class="lbl">акцент</div><div class="h2">42%</div></div>'
                     '<div class="card"><div class="lbl">панель</div><div class="h2"><span class="ser">курсив</span></div>'
                     '<div class="small">приглушённый текст</div></div></div>',
             "anim": [{"sel": ".card", "at": 3.7, "from": {"opacity": 0, "y": 40}, "stagger": 0.15},
                      {"sel": ".card:not(.hot)", "at": 4.8, "to": {"borderColor": "var(--accent-hi)"}}]}]},
        "simple": {"template": "simple", "emph": emph},
        "alternate": {"template": "alternate", "visual": [{"start": 2.0, "end": 5.5}], "emph": emph, "beats": [
            {"kind": "custom", "start": 2.0, "end": 5.5,
             "html": '<div class="col center"><div class="lbl">без лица, только голос</div>'
                     '<div class="card hot"><div class="h1">3-7 c</div></div>'
                     '<div class="card"><div class="h2"><span class="ser">графика</span> на весь кадр</div></div></div>',
             "anim": [{"sel": ".card", "at": 2.1, "from": {"opacity": 0, "y": 60}, "stagger": 0.2}]}]},
    }
    sheets = []
    for name, comp in comps.items():
        save(work / f"{name}.json", comp, 1)
        cmd_compose(NS(work=str(work), composition=str(work / f"{name}.json"), out=str(work / "out" / f"{name}.mp4"),
                       keep_tmp=False))
        sheet = work / "out" / f"{name}.jpg"
        cmd_sheet(NS(video=str(work / "out" / f"{name}.mp4"), n=4, cols=4, width=270, out=str(sheet)))
        sheets.append(sheet)
    tile_sheet(sheets, out / "video.jpg", cols=1, width=1080)
    print(f"preview: {out / 'video.jpg'} (rows: visuals, simple, alternate)"
          + ("" if a.video_only else f", {out / 'carousel.jpg'}"))


# ---------- setup checks ----------

def cmd_doctor(a):
    """Checks every dependency; prints one line per check, exit code 1 if any failed."""
    bad = []

    def check(name, good, hint):
        print(("OK   " if good else "FAIL ") + name + ("" if good else f"\n     -> {hint}"))
        if not good:
            bad.append(name)

    reinstall = "запустите установку ещё раз: bash install.sh"
    for tool in ("ffmpeg", "ffprobe", "node", "npx"):
        check(tool, shutil.which(tool), reinstall)
    if shutil.which("node"):
        v = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        check(f"node {v} (нужна 18+)", re.match(r"v(\d+)", v) and int(re.match(r"v(\d+)", v).group(1)) >= 18, reinstall)
    for mod in ("numpy", "cv2", "faster_whisper", "onnx_asr"):
        try:
            __import__(mod)
            check(f"python: {mod}", True, "")
        except Exception as e:
            check(f"python: {mod}", False, f"{e}; {reinstall}")
    check("модель лиц YuNet", YUNET.exists(), reinstall)
    pj = HYPERFRAMES / "node_modules" / "hyperframes" / "package.json"
    hv = load(pj).get("version") if pj.exists() else None
    check(f"hyperframes {hv or '-'} (нужна {HF_VERSION})", hv == HF_VERSION, reinstall)
    check("иконки lucide-static", LUCIDE.exists(), reinstall)
    if pj.exists():
        r = subprocess.run(["npx", "hyperframes", "browser", "path"], capture_output=True, text=True, cwd=HYPERFRAMES)
        path = (r.stdout.strip().splitlines() or [""])[-1]
        check("Chrome для рендера", r.returncode == 0 and Path(path).exists(), reinstall)
    warn = []
    theme_css(warn)
    check("стиль brand/theme.css", not warn, "; ".join(warn) + " (исправьте через скилл style)")
    free = shutil.disk_usage(ROOT).free / 1e9
    check(f"свободно на диске {free:.0f} ГБ (нужно 5+)", free >= 5, "освободите место на диске")
    print("ВСЁ ГОТОВО" if not bad else f"проблем: {len(bad)}")
    sys.exit(1 if bad else 0)


# ---------- fonts (Fontsource: Google Fonts and more as woff2) ----------

FONTSOURCE_API = "https://api.fontsource.org/v1/fonts/"
FONTSOURCE_CDN = "https://cdn.jsdelivr.net/fontsource/fonts/"
UNICODE_RANGE = {
    "cyrillic": "U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116",
    "latin": "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+20AC, "
             "U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD",
}
FONT_ROLES = {"sans": [400, 600, 800], "accent": [400, 700]}  # weights the templates use


def http_get(url):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "shorts-kit-vid/1.0"})  # the API refuses Python's default UA
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception as e:
        die(f"download failed: {url}: {e}")


def cmd_font(a):
    """Download a font (Fontsource id, e.g. montserrat) into brand/fonts and make it the sans
    (main) or accent font in brand/theme.css: rewrites the fonts:ROLE block and the token."""
    info = json.loads(http_get(FONTSOURCE_API + a.id))
    fam = info.get("family") or die(f"no font '{a.id}' on Fontsource")
    if "cyrillic" not in info.get("subsets", []):
        die(f"{fam} has no Cyrillic: pick another font")
    style = a.style or ("italic" if a.role == "accent" and "italic" in info["styles"] else "normal")
    if style not in info["styles"]:
        die(f"{fam} has no {style} style (has {info['styles']})")
    faces, have = [], info["weights"]
    for want in FONT_ROLES[a.role]:
        w = min(have, key=lambda x: (abs(x - want), -x))  # nearest weight; ties go heavier
        for sub in ("cyrillic", "latin"):
            name = f"{a.id}-{sub}-{w}-{style}.woff2"
            f = BRAND / "fonts" / name
            if not f.exists():
                f.parent.mkdir(exist_ok=True)
                f.write_bytes(http_get(f"{FONTSOURCE_CDN}{a.id}@latest/{sub}-{w}-{style}.woff2"))
            faces.append(f'@font-face {{ font-family: "{fam}"; font-style: {style}; font-weight: {want}; '
                         f'font-display: block; src: url("assets/brand/fonts/{name}") format("woff2"); '
                         f'unicode-range: {UNICODE_RANGE[sub]}; }}')
    theme = BRAND / "theme.css"
    css = theme.read_text()
    block = re.compile(rf"/\* fonts:{a.role} \*/.*?/\* /fonts:{a.role} \*/", re.S)
    if not block.search(css):
        die(f"brand/theme.css has no /* fonts:{a.role} */ ... /* /fonts:{a.role} */ block")
    css = block.sub(lambda m: f"/* fonts:{a.role} */\n" + "\n".join(faces) + f"\n/* /fonts:{a.role} */", css)
    fallback = {"serif": "serif", "handwriting": "cursive"}.get(info.get("category"), "sans-serif")
    token = "--font-sans" if a.role == "sans" else "--font-accent"
    css = re.sub(rf"({token}:\s*)[^;]*;", rf'\g<1>"{fam}", {fallback};', css)
    if a.role == "accent":
        css = re.sub(r"(--accent-style:\s*)[^;]*;", rf"\g<1>{style};", css)
    theme.write_text(css)
    used = set(re.findall(r"assets/brand/fonts/([^\"')]+)", css))
    for f in (BRAND / "fonts").glob("*.woff2"):  # drop files no @font-face points at any more
        if f.name not in used:
            f.unlink()
    print(f"{a.role}: {fam} {style}, weights {sorted({min(have, key=lambda x: (abs(x - w), -x)) for w in FONT_ROLES[a.role]})}"
          f" -> brand/theme.css; check with: tools/vid preview")


def cmd_fetch_models(a):
    """Download the speech models now, so the first edit does not wait for them."""
    import onnx_asr
    from faster_whisper import WhisperModel
    onnx_asr.load_model("gigaam-v3-e2e-rnnt")
    onnx_asr.load_vad("silero")
    print("GigaAM v3 + Silero VAD (русская речь): OK")
    WhisperModel("base", device="cpu", compute_type="int8")
    print("Whisper base (определение языка): OK")
    if a.english:
        WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
        print("Whisper large-v3-turbo (английская речь): OK")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(prog="vid", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("edit", help="the whole v2 flow for one video, resumable: init -> transcribe -> takes -> "
                                     "mistakes -> [cuts.json] -> cut -> [composition.json] -> compose -> sheet")
    s.add_argument("video")
    s.add_argument("-w", "--work")
    s.add_argument("--lang", default="auto")
    s.add_argument("--from", dest="from_step", choices=EDIT_STEPS, help="redo this step and everything after it")
    s.set_defaults(f=cmd_edit)

    s = sub.add_parser("init", help="create a work dir for a source video")
    s.add_argument("video")
    s.add_argument("-w", "--work")
    s.set_defaults(f=cmd_init)

    s = sub.add_parser("transcribe", help="speech recognition -> words.json, transcript.txt")
    s.add_argument("work")
    s.add_argument("--engine", default="auto", choices=["auto", "gigaam", "whisper"],
                   help="auto = gigaam (+Silero VAD) for Russian, whisper otherwise")
    s.add_argument("--min-silence", type=float, default=0.15, help="gigaam: VAD splits speech on pauses this long (s)")
    s.add_argument("--model", default="large-v3-turbo", help="faster-whisper model (small is ~4x faster)")
    s.add_argument("--lang", default="auto", help="en, ru or auto")
    s.add_argument("--prompt", help="vocabulary hint, e.g. 'Claude, RAG, LangGraph'")
    s.set_defaults(f=cmd_transcribe)

    s = sub.add_parser("takes", help="split the transcript into takes, flag finished/mumbled/stutter/retake -> takes.json")
    s.add_argument("work")
    s.add_argument("--edl", help="JSON with {segments: [{start, end}]} to score instead of the transcript segments")
    s.add_argument("--retake-window", type=float, default=60.0,
                   help="a take opening like an earlier one within this many seconds marks the earlier one a retake")
    s.add_argument("--retake-gap", type=float, default=1.0,
                    help="in default (no --edl) mode, sub-split a Whisper segment on any internal silence "
                         "longer than this (s) before scoring - too long to be a mid-sentence breath, "
                         "likely a separate spoken attempt Whisper's VAD didn't break on")
    s.set_defaults(f=cmd_takes)

    s = sub.add_parser("mistakes", help="word-level cut candidates (stutter, retake, false start, profanity) "
                                         "-> mistakes.json; approve them into cuts.json")
    s.add_argument("work")
    s.set_defaults(f=cmd_mistakes)

    s = sub.add_parser("cut", help="kept takes (no retake/profanity/no_words flag) -> out/cut.mp4 + cut.json "
                                    "(segments, words on the cut timeline)")
    s.add_argument("work")
    s.add_argument("--keep", help="comma-separated take numbers to keep despite their flags")
    s.add_argument("--drop", help="comma-separated take numbers to drop")
    s.add_argument("--cuts", help='approved cuts, default WORK/cuts.json: {"drop": [take], "keep": [take], '
                                  '"cuts": [{"start", "end", "reason"}]} in source seconds')
    s.set_defaults(f=cmd_cut)

    s = sub.add_parser("icons", help="Lucide icon names matching a word, for beat icons")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=30)
    s.set_defaults(f=cmd_icons)

    s = sub.add_parser("compose", help="cut.json + composition.json -> out/final.mp4 + .srt via HyperFrames "
                                        "(grid bg, speaker full -> circle -> full, stage beats, caption line)")
    s.add_argument("work")
    s.add_argument("composition", nargs="?", help="default: WORK/composition.json (optional)")
    s.add_argument("-o", "--out")
    s.add_argument("--keep-tmp", action="store_true", help="keep the staged html/clip/variables in hyperframes/.tmp")
    s.set_defaults(f=cmd_compose)

    s = sub.add_parser("spans", help="template alternate: suggested face / visual parts (3-7 s) with their words")
    s.add_argument("work")
    s.add_argument("--min", type=float, default=ALT_SPAN[0])
    s.add_argument("--max", type=float, default=ALT_SPAN[1])
    s.set_defaults(f=cmd_spans)

    s = sub.add_parser("say", help="find when a phrase was spoken -> time and matched words")
    s.add_argument("work")
    s.add_argument("text")
    s.add_argument("--start", type=float, default=0)
    s.add_argument("--end", type=float, default=1e9)
    s.add_argument("--cut", action="store_true", help="search cut.json words (times on the cut timeline)")
    s.set_defaults(f=cmd_say)

    s = sub.add_parser("carousel", help="WORK/slides.json -> WORK/out/NN.png + sheet.jpg + report.json "
                                         "(static slides in the house style, audited by hyperframes check)")
    s.add_argument("work")
    s.add_argument("--keep-tmp", action="store_true", help="keep the staged project in hyperframes/.tmp")
    s.set_defaults(f=cmd_carousel)

    s = sub.add_parser("sheet", help="contact sheet: N downscaled frames tiled in one jpg")
    s.add_argument("video")
    s.add_argument("-n", type=int, default=12)
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--width", type=int, default=240)
    s.add_argument("-o", "--out")
    s.set_defaults(f=cmd_sheet)

    s = sub.add_parser("preview", help="brand/preview/: the current brand style on a sample carousel and video frames")
    s.add_argument("--carousel-only", action="store_true")
    s.add_argument("--video-only", action="store_true")
    s.set_defaults(f=cmd_preview)

    s = sub.add_parser("font", help="download a Fontsource font into brand/fonts and set it in brand/theme.css")
    s.add_argument("id", help="Fontsource id, e.g. montserrat, pt-serif, playfair-display (fontsource.org)")
    s.add_argument("--role", required=True, choices=list(FONT_ROLES), help="sans = main text, accent = emphasis")
    s.add_argument("--style", choices=["normal", "italic"], help="default: italic for accent when available")
    s.set_defaults(f=cmd_font)

    s = sub.add_parser("doctor", help="check that every dependency is installed")
    s.set_defaults(f=cmd_doctor)

    s = sub.add_parser("fetch-models", help="download the speech models now (else on the first transcribe)")
    s.add_argument("--english", action="store_true", help="also Whisper large-v3-turbo (~1.6 GB) for English speech")
    s.set_defaults(f=cmd_fetch_models)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
