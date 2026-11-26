type Cell = string | number | boolean | null;

/** One CSV cell. A text cell that a spreadsheet would read as a formula (starting with = + - @,
 * tab or carriage return, as a file name or class name might) gets a leading ' so it stays text. */
export function csvCell(v: Cell): string {
  let s = v === null ? "" : String(v);
  if (typeof v === "string" && /^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** ``rows`` (the first is the header) as CSV text. */
export function csvText(rows: Cell[][]): string {
  return rows.map((r) => r.map(csvCell).join(",")).join("\n") + "\n";
}

/** Download ``rows`` (the first is the header) as a CSV file named ``name``. */
export function downloadCsv(name: string, rows: Cell[][]) {
  const blob = new Blob([csvText(rows)], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); }, 0);
}
