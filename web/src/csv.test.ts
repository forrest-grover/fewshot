import { describe, expect, it } from "vitest";
import { csvCell, csvText } from "./csv";

describe("csvCell", () => {
  it("quotes cells with commas, quotes and newlines", () => {
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
    expect(csvCell("x\ny")).toBe('"x\ny"');
  });

  it("writes numbers, booleans and null as they are", () => {
    expect(csvCell(-0.25)).toBe("-0.25");
    expect(csvCell(true)).toBe("true");
    expect(csvCell(null)).toBe("");
  });

  it("keeps text that looks like a formula as text", () => {
    expect(csvCell("=HYPERLINK(\"http://x\")")).toBe('"\'=HYPERLINK(""http://x"")"');
    expect(csvCell("+1")).toBe("'+1");
    expect(csvCell("-2+3")).toBe("'-2+3");
    expect(csvCell("@SUM(A1)")).toBe("'@SUM(A1)");
    expect(csvCell("\tcmd")).toBe("'\tcmd");
    expect(csvCell("cat=dog.jpg")).toBe("cat=dog.jpg");
    expect(csvCell("-1e3+A1")).toBe("'-1e3+A1");
  });

  it("leaves negative numbers as numbers, given as text or not", () => {
    expect(csvCell((-0.12345).toFixed(4))).toBe("-0.1235");
    expect(csvCell("-12")).toBe("-12");
    expect(csvCell("-1.5e-7")).toBe("-1.5e-7");
    expect(csvCell(-1.5e-7)).toBe("-1.5e-7");
  });
});

describe("csvText", () => {
  it("joins rows with a final newline", () => {
    expect(csvText([["file", "label"], ["a.jpg", "cat"]])).toBe("file,label\na.jpg,cat\n");
  });
});
