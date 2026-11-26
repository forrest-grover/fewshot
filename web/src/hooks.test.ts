import { afterEach, describe, expect, it } from "vitest";
import { Gate, Latest, stepToast, taskUndoKey, typingAt } from "./hooks";
import { taskDetail } from "./test-data";

/** A keydown dispatched at ``target`` (an element in the document); its ``target`` stays set. */
function press(target: Element, init: KeyboardEventInit): KeyboardEvent {
  const e = new KeyboardEvent("keydown", { bubbles: true, ...init });
  target.dispatchEvent(e);
  return e;
}

function add(html: string): Element {
  const box = document.createElement("div");
  box.innerHTML = html;
  document.body.append(box);
  const el = box.firstElementChild;
  if (!el) throw new Error("no element");
  return el;
}

afterEach(() => { document.body.innerHTML = ""; });

describe("taskUndoKey", () => {
  const ctrlZ = { key: "z", ctrlKey: true };

  it("undoes and redoes from the page", () => {
    expect(taskUndoKey(press(document.body, ctrlZ))).toBe("undo");
    expect(taskUndoKey(press(document.body, { key: "Z", ctrlKey: true, shiftKey: true }))).toBe("redo");
    expect(taskUndoKey(press(document.body, { key: "y", metaKey: true }))).toBe("redo");
    expect(taskUndoKey(press(document.body, { key: "z" }))).toBeNull();
    expect(taskUndoKey(press(document.body, { key: "z", ctrlKey: true, altKey: true }))).toBeNull();
  });

  it("leaves Ctrl+Z to text fields", () => {
    for (const html of ["<input>", "<input type='search'>", "<textarea></textarea>", "<div contenteditable='true'></div>"]) {
      expect(taskUndoKey(press(add(html), ctrlZ)), html).toBeNull();
    }
  });

  it("undoes from a checkbox, select, button, slider or date picker left focused", () => {
    for (const html of ["<input type='checkbox'>", "<select><option>a</option></select>", "<button>x</button>",
      "<input type='range'>", "<input type='color'>", "<input type='date'>"]) {
      expect(taskUndoKey(press(add(html), ctrlZ)), html).toBe("undo");
    }
  });

  it("undoes in the image viewer but not behind another dialog", () => {
    const viewer = add("<div aria-modal='true' data-allows-undo><img></div>");
    expect(taskUndoKey(press(viewer, ctrlZ))).toBe("undo");
    const editor = add("<form aria-modal='true'><button>x</button></form>");
    expect(taskUndoKey(press(editor, ctrlZ))).toBeNull();
  });
});

describe("typingAt", () => {
  it("counts text fields and selects, not checkboxes or buttons", () => {
    expect(typingAt(add("<input>"))).toBe(true);
    expect(typingAt(add("<select></select>"))).toBe(true);
    expect(typingAt(add("<input type='range'>"))).toBe(true); // arrows move it, not page shortcuts
    expect(typingAt(add("<input type='checkbox'>"))).toBe(false);
    expect(typingAt(add("<button></button>"))).toBe(false);
    expect(typingAt(window)).toBe(false);
  });
});

describe("Gate", () => {
  it("ignores calls while one is in flight", async () => {
    const gate = new Gate();
    let release = () => { /* set below */ };
    let calls = 0;
    const slow = () => { calls += 1; return new Promise<number>((res) => { release = () => { res(calls); }; }); };
    const first = gate.run(slow);
    expect(await gate.run(slow)).toBeUndefined(); // a second press while the first is pending
    expect(gate.running).toBe(true);
    release();
    expect(await first).toBe(1);
    expect(gate.running).toBe(false);
    const third = gate.run(slow);
    release();
    expect(await third).toBe(2);
  });

  it("opens again after a failure", async () => {
    const gate = new Gate();
    await expect(gate.run(() => Promise.reject(new Error("x")))).rejects.toThrow("x");
    expect(await gate.run(() => Promise.resolve(1))).toBe(1);
  });
});

describe("Latest", () => {
  it("lets only the newest request through", () => {
    const l = new Latest();
    const first = l.begin();
    expect(first()).toBe(true);
    const second = l.begin();
    expect(first()).toBe(false); // an older response arriving after a newer request began
    expect(second()).toBe(true);
  });
});

describe("stepToast", () => {
  it("names the step from the response", () => {
    const afterUndo = taskDetail({ redo_label: "delete 3 examples", can_redo: true, undo_label: "add class" });
    expect(stepToast("undo", afterUndo, "stale label")).toEqual({ text: "Undid delete 3 examples", then: "redo" });
    const afterRedo = taskDetail({ undo_label: "delete 3 examples", can_undo: true });
    expect(stepToast("redo", afterRedo, "")).toEqual({ text: "Redid delete 3 examples", then: "undo" });
    expect(stepToast("undo", taskDetail({ redo_label: "", can_redo: false }), "")).toEqual({ text: "Undid the last change", then: null });
  });
});
