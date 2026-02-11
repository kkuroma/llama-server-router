import asyncio, aiohttp, random, time, argparse, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

C = dict(base="#1e1e2e", mantle="#181825", surface0="#313244", surface1="#45475a",
         text="#cdd6f4", subtext="#a6adc8", blue="#89b4fa", peach="#fab387",
         green="#a6e3a1", red="#f38ba8", mauve="#cba6f7", yellow="#f9e2af",
         teal="#94e2d5", lavender="#b4befe", overlay0="#6c7086")

plt.rcParams.update({
    "figure.facecolor": C["base"], "axes.facecolor": C["mantle"],
    "axes.edgecolor": C["surface1"], "axes.labelcolor": C["subtext"],
    "axes.titlecolor": C["text"], "xtick.color": C["overlay0"],
    "ytick.color": C["overlay0"], "text.color": C["text"],
    "legend.facecolor": C["surface0"], "legend.edgecolor": C["surface1"],
    "legend.labelcolor": C["text"], "grid.color": C["surface0"], "grid.alpha": 0.6,
    "font.size": 11, "font.family": ["Noto Sans", "DejaVu Sans", "sans-serif"],
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

MODELS = ["GPT-OSS-20b", "GLM-4.7-Flash", "Qwen3-4B-Instruct"]
PALETTE = [C["blue"], C["peach"], C["mauve"], C["teal"], C["yellow"], C["lavender"]]
THINGS = ["avocado","blackhole","cyberpunk","dinosaur","espresso","firewall","glacier",
          "hoverboard","island","jellyfish","kangaroo","lighthouse","monolith","nebula",
          "origami","pyramid","quantum","rainforest","supernova","telescope","volcano",
          "windmill","xenon","yacht","zeppelin"]

async def make_request(session, model, topic, delay, req_id):
    await asyncio.sleep(delay)
    start = time.perf_counter()
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "user", "content": f"Write a 250 word essay about {topic}. Start the essay with the word {topic}"}]}
    try:
        async with session.post("http://localhost:11435/v1/chat/completions", json=payload) as resp:
            j = await resp.json()
            end = time.perf_counter()
            content = j["choices"][0]["message"].get("content", "")
            tokens = j.get("usage", {}).get("completion_tokens", 0)
            dur = end - start
            return dict(id=req_id, model=model, start=start, end=end, duration=dur,
                        tokens=tokens, tps=tokens/dur if dur > 0 else 0,
                        correct=topic.lower() in content.lower(), success=True)
    except Exception:
        return dict(id=req_id, success=False)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num", type=int, default=12)
    parser.add_argument("--mode", choices=["random", "round-robin", "batched"], default="random")
    parser.add_argument("--dir", type=str, default="./stress_test_output")
    parser.add_argument("--name", type=str, default="Stress Test")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    random.seed(args.seed)
    delays = sorted([random.uniform(0, 60) for _ in range(args.num)])
    if delays:
        delays = [d - delays[0] for d in delays]

    req_configs = []
    if args.mode == "batched":
        for i in range(0, args.num, 4):
            m, t, d = random.choice(MODELS), random.choice(THINGS), delays[i]
            for _ in range(min(4, args.num - i)):
                req_configs.append(dict(model=m, topic=t, delay=d))
    else:
        for i in range(args.num):
            m = MODELS[i % len(MODELS)] if args.mode == "round-robin" else MODELS[random.choice(list(range(len(MODELS))))]
            req_configs.append(dict(model=m, topic=random.choice(THINGS), delay=delays[i]))

    print(f"Starting {args.num} requests in '{args.mode}' mode...")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[make_request(session, r["model"], r["topic"], r["delay"], i)
                                         for i, r in enumerate(req_configs)])
    results = sorted([r for r in results if r["success"]], key=lambda x: x["start"])
    t0 = results[0]["start"]
    for r in results:
        r["rel_start"], r["rel_end"] = r["start"] - t0, r["end"] - t0

    models = sorted(set(r["model"] for r in results))
    mcol = {m: PALETTE[i] for i, m in enumerate(models)}
    total_time = max(r["end"] for r in results) - t0
    agg_tps = sum(r["tokens"] for r in results) / total_time
    correct_pct = sum(r["correct"] for r in results) / len(results) * 100
    avg_dur = sum(r["duration"] for r in results) / len(results)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{args.name}  ·  {len(results)} requests  ·  {args.mode}",
                 fontsize=16, fontweight="bold", color=C["text"], y=0.98)
    xp = np.arange(1, len(results) + 1)
    legend_h = [Patch(facecolor=mcol[m], label=m) for m in models]

    for i, r in enumerate(results):
        ax[0,0].bar(xp[i], r["tps"], color=mcol[r["model"]], width=0.7, edgecolor="none")
    ax[0,0].set(title=f"TPS per Request  (Agg: {agg_tps:.2f})", xlabel="Request ID", ylabel="Tokens / Second")
    ax[0,0].set_xticks(xp); ax[0,0].legend(handles=legend_h, framealpha=0.8); ax[0,0].grid(axis="y", ls="--")

    by_end = sorted(results, key=lambda r: r["rel_end"])
    xpe = np.arange(1, len(by_end) + 1)
    for i, r in enumerate(by_end):
        ax[0,1].bar(xpe[i], r["rel_end"]-r["rel_start"], bottom=r["rel_start"],
                     color=mcol[r["model"]], width=0.7, edgecolor="none")
    ax[0,1].set(title=f"Request Lifespan  (avg {avg_dur:.1f}s)", xlabel="Request (by end time)", ylabel="Time (s)")
    ax[0,1].set_xticks(xpe); ax[0,1].set_xticklabels([str(r["id"]) for r in by_end], rotation=45, ha="right")
    ax[0,1].legend(handles=legend_h, loc="upper left", framealpha=0.8); ax[0,1].grid(axis="y", ls="--")

    events = sorted([(r["rel_start"], 1) for r in results] + [(r["rel_end"], -1) for r in results])
    tp, qa, ra, cq, cr = [0], [0], [0], 0, 0
    for t, d in events:
        tp.append(t); qa.append(cq); ra.append(cr)
        if d == 1: cq += 1
        else: cq -= 1; cr += 1
        tp.append(t); qa.append(cq); ra.append(cr)
    tp, qa, ra = np.array(tp), np.array(qa), np.array(ra)
    auc = np.sum(np.diff(tp) * (qa[:-1] + qa[1:]) / 2.0)
    ax[1,0].fill_between(tp, 0, qa, step="post", color=C["blue"], alpha=0.55, label="In-flight")
    ax[1,0].fill_between(tp, qa, qa+ra, step="post", color=C["red"], alpha=0.45, label="Completed")
    ax[1,0].set(title=f"Queue Load  (AUC: {auc:.1f})", xlabel="Time (s)", ylabel="Count")
    ax[1,0].legend(loc="upper left", framealpha=0.8); ax[1,0].grid(axis="y", ls="--")

    colors = [C["green"] if r["correct"] else C["red"] for r in results]
    ax[1,1].bar(xp, [1]*len(results), color=colors, edgecolor="none")
    ax[1,1].set(title=f"Correctness  ({correct_pct:.0f}%)", xlabel="Request ID"); ax[1,1].set_yticks([])
    ax[1,1].set_xticks(xp)
    ax[1,1].legend(handles=[Patch(facecolor=C["green"], label="Correct"),
                             Patch(facecolor=C["red"], label="Wrong")], loc="upper right", framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(args.dir, "stress_test_results.png")
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Plot saved to {plot_path}")

    with open(os.path.join(args.dir, "report.txt"), "w") as f:
        for r in results:
            line = f"[{r['rel_start']:6.2f}s] ID {r['id']}: {r['model']} | {r['tps']:6.2f} tps | Correct: {r['correct']}"
            print(line); f.write(line + "\n")

if __name__ == "__main__":
    asyncio.run(main())