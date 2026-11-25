/** Download ``rows`` (the first is the header) as a CSV file named ``name``. */
export function downloadCsv(name: string, rows: (string | number | boolean | null)[][]) {
  const cell = (v: string | number | boolean | null) => {
    const s = v === null ? "" : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const blob = new Blob([rows.map((r) => r.map(cell).join(",")).join("\n") + "\n"], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); }, 0);
}
