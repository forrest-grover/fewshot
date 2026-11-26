import { useState } from "react";
import { api } from "./api";
import { ErrorBox } from "./ui";

/** Asks for the server's token (set with `fewshot serve --token`); the server keeps it in a cookie. */
export default function SignIn() {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <main className="sign-in">
      <form
        className="new-task"
        onSubmit={(e) => {
          e.preventDefault();
          setBusy(true);
          api.login(token).then(
            () => { location.reload(); },
            (err: unknown) => { setBusy(false); setError(err instanceof Error ? err.message : String(err)); },
          );
        }}
      >
        <h1>Sign in</h1>
        <p className="muted">This fewshot server needs its access token (the one it was started with, <code>fewshot serve --token</code>).</p>
        <ErrorBox error={error} onClose={() => { setError(null); }} />
        <label>
          <span>Token</span>
          <input type="password" autoComplete="current-password" value={token} onChange={(e) => { setToken(e.target.value); }} required autoFocus />
        </label>
        <button type="submit" disabled={busy || !token}>{busy ? "Signing in…" : "Sign in"}</button>
      </form>
    </main>
  );
}
