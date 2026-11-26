import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { ClassSelect, DropZone } from "./ui";

afterEach(cleanup);

const options = () => Array.from(screen.getByRole<HTMLSelectElement>("combobox").options).map((o) => o.textContent);

describe("ClassSelect", () => {
  it("holds only the current class until used, then all of them", () => {
    const onPick = vi.fn();
    render(<ClassSelect value="dog" classes={["cat", "dog", "fox"]} aria-label="Move to class" onPick={onPick} />);
    expect(options()).toEqual(["dog"]);
    fireEvent.mouseDown(screen.getByRole("combobox"));
    expect(options()).toEqual(["cat", "dog", "fox"]);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "fox" } });
    expect(onPick).toHaveBeenCalledWith("fox");
  });

  it("numbers classes by their place, skipping one excluded, under a placeholder", () => {
    render(<ClassSelect classes={["cat", "dog", "fox"]} numbered exclude="dog" placeholder="Add as other…" aria-label="Add as" onPick={vi.fn()} />);
    expect(options()).toEqual(["Add as other…"]);
    fireEvent.focus(screen.getByRole("combobox"));
    expect(options()).toEqual(["Add as other…", "1 · cat", "3 · fox"]);
  });
});

describe("DropZone", () => {
  it("stays highlighted while a drag moves over its children", () => {
    render(<DropZone onFiles={vi.fn()}><span>inside</span></DropZone>);
    const zone = screen.getByRole("button");
    const files = { types: ["Files"] };
    fireEvent.dragOver(zone, { dataTransfer: files });
    expect(zone.className).toContain("over");
    // jsdom has no DragEvent, and a plain Event drops relatedTarget: use a MouseEvent.
    const leave = (to: Element) => { fireEvent(zone, new MouseEvent("dragleave", { bubbles: true, relatedTarget: to })); };
    leave(screen.getByText("inside")); // onto a child
    expect(zone.className).toContain("over");
    leave(document.body); // out of it
    expect(zone.className).not.toContain("over");
  });
});
