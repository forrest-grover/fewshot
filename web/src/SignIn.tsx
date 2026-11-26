import { useRef, useState } from "react";
import { api } from "./api";
import { useFocusTrap } from "./keys";
import { ErrorBox } from "./ui";

/** Asks for the server's token (set with `fewshot serve --token`); the server keeps it in a cookie.
 * A dialog over the app, so a session that ends mid-use keeps everything shown (prediction results,
 * evaluations, the page itself) for when ``onSignedIn`` is called. */
export default function SignIn({ again, onSignedIn }: { again: boolean; onSignedIn: () => void }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useFocusTrap(box);
  return (
    <div className="modal-backdrop">
      <div ref={box} className="modal sign-in" role="dialog" aria-modal="true" aria-labelledby="sign-in-title">
        <form
          className="col"
          onSubmit={(e) => {
            e.preventDefault();
            setBusy(true);
            api.login(token).then(
              () => { setBusy(false); setToken(""); onSignedIn(); },
              (err: unknown) => { setBusy(false); setError(err instanceof Error ? err.message : String(err)); },
            );
          }}
        >
          <h3 className="modal-title" id="sign-in-title">Sign in</h3>
          <p className="muted">
            {again ? "Your session has ended (it lasts 30 days, and ends when you log out or the server's token changes). " : ""}
            This fewshot server needs its access token (the one it was started with, <code>fewshot serve --token</code>).
          </p>
          <ErrorBox error={error} onClose={() => { setError(null); }} />
          <label>
            <span>Token</span>
            <input type="password" autoComplete="current-password" value={token} onChange={(e) => { setToken(e.target.value); }} required autoFocus />
          </label>
          <button type="submit" disabled={busy || !token}>{busy ? "Signing in…" : "Sign in"}</button>
        </form>
      </div>
    </div>
  );
}
