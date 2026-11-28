// How numbers are shown, the same on every tab.

/** A rate such as accuracy or recall, as a percentage with one decimal ("97.5%"). */
export const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

/** A probability (a prediction's relative score): two significant digits, and ">99%" or "<1%" at
 * the ends, so an image that fits no class well never shows as "100.0%". */
export function prob(v: number | null | undefined): string {
  if (v == null) return "–";
  const p = v * 100;
  if (p > 99) return ">99%";
  if (p < 1) return p <= 0 ? "0%" : "<1%";
  return `${String(Number(p.toPrecision(2)))}%`;
}

/** A cosine similarity, with two decimals ("0.71"). Fields that set a threshold keep three. */
export const sim = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(2));

/** A threshold value, with three decimals. */
export const thr = (v: number) => v.toFixed(3);

/** What the similarity and the relative score mean, for tooltips. */
export const SIM_HELP = "Cosine similarity to the closest class (its examples, or its description): low values mean the image fits none of the classes well.";
export const SCORE_HELP = "Relative score: compares the known classes only, so it is high even for an image that belongs to none of them. "
  + "It is not calibrated unless a temperature is set (Settings).";
export const REJECT_HELP = "Images whose similarity to every class is below the reject threshold are called “unknown” (API and CLI: reject_below).";
