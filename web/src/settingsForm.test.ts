import { describe, expect, it } from "vitest";
import { afterSave, formOf, pickClassifier, settingsPatch } from "./settingsForm";
import { taskDetail } from "./test-data";

describe("settingsPatch", () => {
  const fitted = taskDetail({ reject_below: 0.7, reject_below_zero_shot: 0.22, temperature: 0.05, classifier: "knn",
    classifier_params: { k: 5 } });

  it("sends nothing for an untouched form", () => {
    expect(settingsPatch(fitted, formOf(fitted))).toEqual({ patch: {}, problems: [] });
  });

  it("switching embedder sends only the embedder, so the server clears values fitted to the old one", () => {
    const { patch } = settingsPatch(fitted, { ...formOf(fitted), embedder: "timm:other" });
    expect(patch).toEqual({ embedder: "timm:other" });
  });

  it("treats the same number typed differently as unchanged", () => {
    expect(settingsPatch(fitted, { ...formOf(fitted), reject: " 0.70", params: { k: "5.0" } }).patch).toEqual({});
  });

  it("sends a changed threshold, and clears one blanked", () => {
    const { patch } = settingsPatch(fitted, { ...formOf(fitted), reject: "0.65", temperature: "", rejectZero: "0.22" });
    expect(patch).toEqual({ reject_below: 0.65, clear_temperature: true });
  });

  it("doesn't clear what is already off", () => {
    const t = taskDetail();
    expect(settingsPatch(t, { ...formOf(t), reject: "", temperature: "  " }).patch).toEqual({});
  });

  it("leaves a new classifier's params to the server unless some are typed", () => {
    expect(settingsPatch(fitted, { ...formOf(fitted), classifier: "prototype", params: {} }).patch).toEqual({ classifier: "prototype" });
    expect(settingsPatch(fitted, { ...formOf(fitted), classifier: "prototype", params: { scale: "20" } }).patch)
      .toEqual({ classifier: "prototype", classifier_params: { scale: 20 } });
  });

  it("sends the whole params dict when one is changed or blanked", () => {
    expect(settingsPatch(fitted, { ...formOf(fitted), params: { k: "7", scale: "" } }).patch).toEqual({ classifier_params: { k: 7 } });
    expect(settingsPatch(fitted, { ...formOf(fitted), params: { k: "" } }).patch).toEqual({ classifier_params: {} });
  });

  it("reports what can't be parsed and doesn't send it", () => {
    const r = settingsPatch(fitted, { ...formOf(fitted), reject: "abc", temperature: "500", params: { k: "x" } });
    expect(r.problems).toHaveLength(3);
    expect(r.patch).toEqual({ classifier_params: {} });
  });
});

describe("pickClassifier", () => {
  const saved = taskDetail({ classifier: "knn", classifier_params: { k: 7 } });

  it("starts another classifier from its defaults", () => {
    expect(pickClassifier({ flip: true }, saved, "prototype")).toEqual({ flip: true, classifier: "prototype", params: {} });
  });

  it("going back to the saved classifier keeps its saved params", () => {
    const away = pickClassifier({ reject: "0.5" }, saved, "prototype");
    const back = pickClassifier(away, saved, "knn");
    expect(back).toEqual({ reject: "0.5" });
    expect(settingsPatch(saved, { ...formOf(saved), ...back }).patch).toEqual({ reject_below: 0.5 });
  });
});

describe("afterSave", () => {
  it("drops the edits that were saved and keeps ones typed meanwhile", () => {
    const sent = { reject: "0.5", flip: true };
    expect(afterSave({ reject: "0.55", flip: true, names: false }, sent)).toEqual({ reject: "0.55", names: false });
  });
});
