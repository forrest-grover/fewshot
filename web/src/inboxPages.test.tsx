import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type InboxEntry } from "./api";
import type { Uploads } from "./useUploads";
import Inbox from "./Inbox";
import { fetchInbox } from "./inboxPages";
import { taskDetail } from "./test-data";

afterEach(cleanup);

const entry = (i: number): InboxEntry => ({
  id: `i${String(i)}`, source: `${String(i)}.jpg`, file: `${String(i)}.jpg`, added: "2026-01-01T00:00:00", margin: null, prediction: null,
});
/** A server holding ``n`` inbox images. */
const server = (n: number) => vi.fn((offset: number, limit: number) => Promise.resolve({
  items: Array.from({ length: Math.max(0, Math.min(limit, n - offset)) }, (_, k) => entry(offset + k)), total: n, offset,
}));

describe("fetchInbox", () => {
  it("fetches pages until it has as many as wanted", async () => {
    const page = server(2500);
    const r = await fetchInbox(page, 1500, 1000);
    expect(page.mock.calls).toEqual([[0, 1000], [1000, 1000]]);
    expect(r?.items).toHaveLength(2000);
    expect(r?.total).toBe(2500);
  });

  it("fetches one page when fewer are wanted, and stops at the end", async () => {
    const one = server(2500);
    await fetchInbox(one, 0, 1000);
    expect(one).toHaveBeenCalledTimes(1);
    const short = server(1200);
    expect((await fetchInbox(short, 5000, 1000))?.items).toHaveLength(1200);
    expect(short).toHaveBeenCalledTimes(2);
  });

  it("gives up if a request fails", async () => {
    expect(await fetchInbox(() => Promise.resolve(undefined), 10, 5)).toBeUndefined();
  });
});

describe("Inbox", () => {
  it("keeps the pages loaded with Load more when the task changes", async () => {
    const page = server(1500);
    const inbox = vi.spyOn(api, "inbox").mockImplementation((_n, _s, offset = 0, limit = 1000) => page(offset, limit));
    const uploads: Uploads = { slots: {}, start: () => Promise.resolve([]), dismiss: vi.fn() };
    const props = { setTask: vi.fn(), run: <T,>(p: Promise<T>) => p, uploads };
    const { rerender } = render(<Inbox task={taskDetail({ inbox_count: 1500 })} {...props} />);
    await screen.findByText(/1000 loaded/);
    // Show every card loaded, then the next page.
    for (let i = 0; i < 4; i++) fireEvent.click(screen.getByText(/Show more/));
    fireEvent.click(await screen.findByText(/Load more/));
    await waitFor(() => { expect(screen.queryByText(/loaded;/)).toBeNull(); }); // all 1500 loaded
    inbox.mockClear();
    rerender(<Inbox task={taskDetail({ inbox_count: 1500, revision: 4 })} {...props} />);
    await waitFor(() => { expect(inbox).toHaveBeenCalledTimes(2); });
    expect(inbox.mock.calls.map((c) => c[2])).toEqual([0, 1000]);
    await new Promise((r) => setTimeout(r, 10));
    expect(screen.queryByText(/Load more/)).toBeNull();
  });
});
