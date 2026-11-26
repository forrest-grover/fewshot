import { useState } from "react";
import type { HeldOut } from "./api";

const BINS = 24;
const W = 560, H = 150, PAD = { l: 34, r: 8, t: 8, b: 24 };

interface Props {
  predictions: HeldOut[];
  current: number | null; // the task's reject threshold
  suggested: number | null;
  onSet: (value: number) => void;
}

/** Histogram of held-out similarities, correct and wrong predictions stacked, with the current and
 * suggested reject thresholds marked. Clicking a bar picks its lower edge as a threshold to apply.
 * From the keyboard the chart is one control (a single tab stop): arrows (and Home, End) move
 * between bars, Enter or Space picks the one shown. */
export default function Threshold({ predictions, current, suggested, onSet }: Props) {
  // The threshold typed or picked from a bar, as typed (so the field can be cleared); null: the suggestion.
  const [text, setText] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null); // the bar under the pointer
  const [active, setActive] = useState<number | null>(null); // the bar chosen with the keyboard, while focused
  if (predictions.length === 0) return null;

  const sims = predictions.map((p) => p.similarity);
  const marks = [current, suggested].filter((v): v is number => v !== null);
  let lo = Math.min(...sims, ...marks), hi = Math.max(...sims, ...marks);
  if (hi - lo < 1e-3) { lo -= 0.01; hi += 0.01; }
  const step = (hi - lo) / BINS;
  const bins = Array.from({ length: BINS }, (_, i) => ({ from: lo + i * step, correct: 0, wrong: 0 }));
  for (const p of predictions) {
    const b = bins[Math.min(BINS - 1, Math.floor((p.similarity - lo) / step))];
    if (p.correct) b.correct++; else b.wrong++;
  }
  const top = Math.max(1, ...bins.map((b) => b.correct + b.wrong));
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
  const x = (v: number) => PAD.l + ((v - lo) / (hi - lo)) * plotW;
  const y = (n: number) => PAD.t + plotH - (n / top) * plotH;
  const bw = plotW / BINS;

  const picked = text === null ? null : text.trim() === "" ? NaN : Number(text); // NaN: empty or not a number
  const value = picked ?? suggested;
  const below = (ok: boolean) => predictions.filter((p) => p.correct === ok && value !== null && p.similarity < value).length;
  const nCorrect = predictions.filter((p) => p.correct).length;
  const nWrong = predictions.length - nCorrect;
  const fmt = (v: number) => v.toFixed(3);
  const pick = (i: number) => { setText(String(Number(bins[i].from.toFixed(3)))); };
  const describe = (i: number) => `${fmt(bins[i].from)} to ${fmt(bins[i].from + step)}: ${String(bins[i].correct)} correct, ${String(bins[i].wrong)} wrong`;
  const shown = hover ?? active; // the bar whose tooltip shows

  return (
    <div className="threshold">
      <div className="legend" aria-hidden="true">
        <span><i className="swatch correct" /> correct</span>
        <span><i className="swatch wrong" /> wrong</span>
        {current !== null && <span><i className="swatch line" /> current threshold</span>}
        {suggested !== null && <span><i className="swatch line dashed" /> suggested</span>}
      </div>
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${String(W)} ${String(H)}`} role="slider" tabIndex={0}
          aria-label={`Similarity of held-out predictions: ${String(nCorrect)} correct, ${String(nWrong)} wrong, from ${fmt(Math.min(...sims))} to ${fmt(Math.max(...sims))}. Arrow keys choose a bar, Enter uses its lower edge as the threshold`}
          aria-valuemin={0} aria-valuemax={BINS - 1} aria-valuenow={active ?? 0} aria-valuetext={describe(active ?? 0)}
          onFocus={() => { setActive((a) => a ?? Math.max(0, bins.findIndex((b) => b.correct + b.wrong > 0))); }}
          onBlur={() => { setActive(null); }}
          onKeyDown={(e) => {
            const a = active ?? 0;
            const to: Record<string, number> = { ArrowLeft: a - 1, ArrowDown: a - 1, ArrowRight: a + 1, ArrowUp: a + 1, Home: 0, End: BINS - 1 };
            if (e.key in to) { e.preventDefault(); setActive(Math.max(0, Math.min(BINS - 1, to[e.key]))); }
            else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(a); }
          }}>
          <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={PAD.t + plotH} y2={PAD.t + plotH} />
          <text className="tick" x={PAD.l - 4} y={PAD.t + 8} textAnchor="end">{top}</text>
          <text className="tick" x={PAD.l - 4} y={PAD.t + plotH} textAnchor="end">0</text>
          {[lo, (lo + hi) / 2, hi].map((v, i) => (
            <text key={i} className="tick" x={x(v)} y={H - 6} textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}>{fmt(v)}</text>
          ))}
          {bins.map((b, i) => {
            const bx = PAD.l + i * bw + 1, w = Math.max(1, bw - 2); // 2px gap between bars
            const hc = (b.correct / top) * plotH, hw = (b.wrong / top) * plotH;
            return (
              <g key={i} className={shown === i ? "bin hover" : "bin"}
                onMouseEnter={() => { setHover(i); }} onMouseLeave={() => { setHover(null); }}
                onClick={() => { pick(i); }}>
                <rect className="hit" x={PAD.l + i * bw} y={PAD.t} width={bw} height={plotH} />
                {b.correct > 0 && <rect className="correct" x={bx} y={y(b.correct)} width={w} height={Math.max(1, hc)} rx={2} />}
                {b.wrong > 0 && <rect className="wrong" x={bx} y={y(b.correct + b.wrong)} width={w} height={Math.max(1, hw - (b.correct ? 2 : 0))} rx={2} />}
              </g>
            );
          })}
          {current !== null && <line className="mark" x1={x(current)} x2={x(current)} y1={PAD.t} y2={PAD.t + plotH} />}
          {suggested !== null && <line className="mark dashed" x1={x(suggested)} x2={x(suggested)} y1={PAD.t} y2={PAD.t + plotH} />}
          {picked !== null && Number.isFinite(picked) && <line className="mark picked" x1={x(picked)} x2={x(picked)} y1={PAD.t} y2={PAD.t + plotH} />}
        </svg>
        {shown !== null && (
          <div className="tooltip" aria-hidden="true" style={{ left: `${String((x(bins[shown].from + step / 2) / W) * 100)}%` }}>
            <b>{fmt(bins[shown].from)}–{fmt(bins[shown].from + step)}</b>
            <div>{bins[shown].correct} correct · {bins[shown].wrong} wrong</div>
            <div className="muted">{hover === null ? "Enter" : "Click"} to use {fmt(bins[shown].from)}</div>
          </div>
        )}
      </div>
      <div className="row">
        <label className="inline">
          <span>Threshold</span>
          <input type="number" min={-1} max={1} step={0.001} className="threshold-input"
            value={text ?? (suggested === null ? "" : String(suggested))} placeholder="pick a bar"
            onChange={(e) => { setText(e.target.value); }} />
        </label>
        {value !== null && Number.isFinite(value) && (
          <span>
            {picked === null ? "(suggested) " : ""}{below(true)} of {nCorrect} correct and {below(false)} of {nWrong} wrong
            predictions would become “unknown”.
          </span>
        )}
      </div>
      {value !== null && Number.isFinite(value) && value >= -1 && value <= 1 && (
        <div className="row">
          <button className="small" disabled={value === current} onClick={() => { onSet(value); }}>Set reject threshold to {fmt(value)}</button>
        </div>
      )}
    </div>
  );
}
