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
