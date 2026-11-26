import type { InboxEntry } from "./api";

interface Page { items: InboxEntry[]; total: number }

/** The first ``want`` images of an inbox list (at least one page, at most all of them), fetched
 * with ``page`` a page of ``pageSize`` at a time: so reloading the list after a change keeps the
 * pages "Load more" added. Undefined if a request fails. */
export async function fetchInbox(
  page: (offset: number, limit: number) => Promise<Page | undefined>, want: number, pageSize: number,
): Promise<Page | undefined> {
  const items: InboxEntry[] = [];
  let total = 0;
  do {
    const r = await page(items.length, pageSize);
    if (!r) return undefined;
    items.push(...r.items);
    total = r.total;
    if (r.items.length < pageSize) break; // the end (the list may have shrunk meanwhile)
  } while (items.length < Math.min(want, total));
  return { items, total };
}

/** Inbox images acted on (accepted or dismissed) but maybe still in the list as last fetched:
 * each id with the task revision its change made, or null while that change is still being saved.
 * They stay hidden until a list fetched at that revision or later arrives. */
export type Gone = ReadonlyMap<string, number | null>;

/** What of ``gone`` still needs hiding once a list fetched at revision ``rev`` arrives: changes
 * still being saved, and saved ones the list may predate. One the list already reflects is
 * forgotten, so the image shows again if an undo brings it back. */
export function stillGone(gone: Gone, rev: number): Gone {
  return new Map([...gone].filter(([, r]) => r === null || r > rev));
}
