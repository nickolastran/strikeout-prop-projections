// K-prop board web UI: one button -> runs src.daily_projections for today,
// then shows the generated HTML board. Zero npm dependencies.
const http = require("http");
const { execFile } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const PY = path.join(ROOT, ".venv", "bin", "python");
const OUT = path.join(ROOT, "outputs", "projections");
const PORT = process.env.PORT || 3000;

const today = () => new Date().toLocaleDateString("en-CA"); // local YYYY-MM-DD

let running = false; // ponytail: single global run flag; fine for a one-user local tool

const PAGE = `<!doctype html>
<meta charset="utf-8">
<title>Strikeout Projections</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; }
  button { font-size: 1.1rem; padding: .6rem 1.4rem; cursor: pointer; }
  button:disabled { cursor: wait; opacity: .6; }
  #status { margin-left: 1rem; color: #555; }
  iframe { width: 100%; height: 80vh; border: 1px solid #ccc; margin-top: 1rem; }
</style>
<h1>Daily Strikeout Projections</h1>
<p>
  <button id="go">Generate today's projections</button>
  <span id="status"></span>
</p>
<iframe id="board" hidden></iframe>
<script>
  const btn = document.getElementById("go");
  const status = document.getElementById("status");
  const board = document.getElementById("board");

  function show(date) {
    board.src = "/board/" + date;
    board.hidden = false;
  }

  // If today's board already exists, show it immediately.
  fetch("/board/today", { method: "HEAD" }).then(r => {
    if (r.ok) { show("today"); status.textContent = "Showing existing board for today."; }
  });

  btn.onclick = async () => {
    btn.disabled = true;
    status.textContent = "Generating\\u2026 (can take a minute)";
    try {
      const r = await fetch("/generate", { method: "POST" });
      const j = await r.json();
      if (j.ok) { status.textContent = "Done \\u2014 " + j.date; show(j.date); }
      else { status.textContent = "Failed: " + j.error; }
    } catch (e) {
      status.textContent = "Failed: " + e;
    }
    btn.disabled = false;
  };
</script>
`;

const server = http.createServer((req, res) => {
  const url = req.url.split("?")[0];

  if (url === "/") {
    res.writeHead(200, { "content-type": "text/html" });
    return res.end(PAGE);
  }

  const m = url.match(/^\/board\/(today|\d{4}-\d{2}-\d{2})$/);
  if (m) {
    const date = m[1] === "today" ? today() : m[1];
    const file = path.join(OUT, `strikeouts_${date}.html`);
    if (!fs.existsSync(file)) {
      res.writeHead(404);
      return res.end("No board for " + date);
    }
    res.writeHead(200, { "content-type": "text/html" });
    return req.method === "HEAD" ? res.end() : res.end(fs.readFileSync(file));
  }

  if (url === "/generate" && req.method === "POST") {
    if (running) {
      res.writeHead(409, { "content-type": "application/json" });
      return res.end(JSON.stringify({ ok: false, error: "already running" }));
    }
    running = true;
    const date = today();
    execFile(PY, ["-m", "src.daily_projections", date],
      { cwd: ROOT, timeout: 10 * 60 * 1000 },
      (err, stdout, stderr) => {
        running = false;
        res.writeHead(200, { "content-type": "application/json" });
        if (err || !fs.existsSync(path.join(OUT, `strikeouts_${date}.html`))) {
          const error = (stderr || String(err) || "unknown error").trim().split("\n").slice(-3).join(" ");
          return res.end(JSON.stringify({ ok: false, error }));
        }
        res.end(JSON.stringify({ ok: true, date }));
      });
    return;
  }

  res.writeHead(404);
  res.end("Not found");
});

server.listen(PORT, () =>
  console.log(`Strikeout projections UI -> http://localhost:${PORT}`));
