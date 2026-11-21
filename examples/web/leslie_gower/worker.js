// The Leslie-Gower page's solves. serve(port) answers each message on port.
// The load unpacks Pyodide 0.28.3, NumPy, the Pyomo 6.10.1 wheel and pounce's
// two wheels for Pyodide, pounce_solver and pyomo_pounce, all inside the page
// file. It starts Pyodide, installs the wheels, writes SiNDAE's sindae
// package and sindae_page.py into Pyodide's files, and imports
// sindae_page.py. A step message runs that step's Python function with the
// step's options. pyomo-pounce solves each model in this process, and while
// the training solves, sindae_page.py sends each iteration's row to the page
// through postProgress here.
//
// The page runs it in a worker made from a data URL with this file, and in
// the page where the browser refuses the worker.

const PYODIDE_VERSION = "0.28.3";
const PYOMO_VERSION = "6.10.1";
const POUNCE_WHEEL = "pounce_solver-0.17.0-cp39-abi3-emscripten_4_0_9_wasm32.whl";
const PYOMO_POUNCE_WHEEL = "pyomo_pounce-0.17.0-py3-none-any.whl";

/**
 * The address Pyodide is told its files are at. No request goes there:
 * while Pyodide starts, fetch answers each address under it with the
 * file of that name from the page file.
 */
const PYODIDE_URL = "https://pyodide.invalid/";

/** Answers the page's load and step messages on `port`. */
function serve(port) {
  let pyodide = null;
  // The message being answered, which the training's rows go to.
  let current = 0;

  // The bytes of a file the page file carries gzipped and in base64.
  async function unpacked(text) {
    const gzipped = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
    return new Uint8Array(await new Response(new Response(gzipped).body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  }

  async function load(packed, python) {
    const files = {};
    for (const [name, text] of Object.entries(packed)) files[name] = await unpacked(text);
    const textOf = (name) => new TextDecoder().decode(files[name]);
    // pyodide.asm.js runs here as a script, which defines
    // _createPyodideModule, so loadPyodide does not load it. The lock file
    // goes to loadPyodide whole, and fetch answers the wasm, the standard
    // library, NumPy's wheel and pounce's wheel from the files while
    // Pyodide starts and loads them.
    (0, eval)(textOf("pyodide.asm.js"));
    const fetched = globalThis.fetch;
    globalThis.fetch = (input, init) => {
      const url = String(input instanceof Request ? input.url : input);
      if (!url.startsWith(PYODIDE_URL)) return fetched(input, init);
      const name = url.slice(PYODIDE_URL.length);
      if (!(name in files)) return Promise.reject(new Error(`the page file has no ${name}`));
      const type = name.endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
      return Promise.resolve(new Response(files[name], { headers: { "Content-Type": type } }));
    };
    try {
      const { loadPyodide } = await import(`data:text/javascript;charset=utf-8,${encodeURIComponent(textOf("pyodide.mjs"))}`);
      pyodide = await loadPyodide({
        indexURL: PYODIDE_URL,
        lockFileContents: JSON.parse(textOf("pyodide-lock.json")),
        stdLibURL: `${PYODIDE_URL}python_stdlib.zip`,
        packageBaseUrl: PYODIDE_URL,
      });
      const problems = [];
      const quiet = { messageCallback: () => {}, errorCallback: (text) => problems.push(text) };
      await pyodide.loadPackage("numpy", quiet);
      // pounce's wheel carries compiled code, which loadPackage loads.
      await pyodide.loadPackage(`${PYODIDE_URL}${POUNCE_WHEEL}`, quiet);
      if (problems.length) throw new Error(`NumPy or pounce did not load: ${problems.join(" ")}`);
    } finally {
      globalThis.fetch = fetched;
    }
    // Pyomo's and pyomo-pounce's wheels have no compiled code, so each is unpacked into site-packages as it stands.
    const sitePackages = pyodide.runPython('import sysconfig; sysconfig.get_paths()["purelib"]');
    for (const name of [`pyomo-${PYOMO_VERSION}-py3-none-any.whl`, PYOMO_POUNCE_WHEEL]) pyodide.unpackArchive(files[name], "zip", { extractDir: sitePackages });
    for (const [name, text] of Object.entries(python)) {
      const path = `/home/pyodide/${name}`;
      pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      pyodide.FS.writeFile(path, text);
    }
    globalThis.postProgress = (text) => port.postMessage({ id: current, progress: JSON.parse(text) });
    pyodide.runPython(`
import importlib, logging, sys
sys.path.insert(0, "/home/pyodide")
importlib.invalidate_caches()
logging.disable(logging.WARNING)
import js
import sindae_page
sindae_page.PROGRESS = js.postProgress
`);
    return files[POUNCE_WHEEL].byteLength;
  }

  port.onmessage = async (event) => {
    const message = event.data;
    const started = performance.now();
    try {
      if (message.kind === "load") {
        const bytes = await load(message.pyodide, message.python);
        port.postMessage({ id: message.id, ok: true, loadMs: performance.now() - started, bytes, pyodide: PYODIDE_VERSION, pyomo: PYOMO_VERSION });
        return;
      }
      current = message.id;
      const step = pyodide.globals.get("sindae_page")[message.kind];
      const answer = JSON.parse(step(JSON.stringify(message.options)));
      port.postMessage({ id: message.id, ok: true, ms: performance.now() - started, answer });
    } catch (error) {
      port.postMessage({ id: message.id, ok: false, message: error.message });
    }
  };
}
