// Builds the Leslie-Gower browser page as one HTML file:
//
//   node examples/web/leslie_gower/build.mjs [out.html]
//
// The default output is examples/web/leslie_gower/build/leslie-gower.html.
// It downloads Pyodide 0.28.3 with its NumPy wheel and the Pyomo 6.10.1
// wheel, each checked against its SHA-256, and writes into the page every
// file the page runs: those files, pounce's two wheels for Pyodide from
// pounce/, the Python files of this repository's sindae package, and
// sindae_page.py. The downloads stay in build/downloads and are used again
// by the next build.
//
// pounce/ holds pounce_solver, pounce's Python package compiled for Pyodide
// 0.28.3, and pyomo_pounce, its Pyomo interface. To move to a new pounce
// release, copy its Pyodide wheel from crates/pounce-wasm/web-python/wheels/
// in pounce's repository, build its pyomo-pounce wheel with
// pip wheel --no-deps pyomo-pounce/, put both in pounce/, and change their
// names in POUNCE here and in worker.js.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..", "..");
const build = path.join(here, "build");
const downloads = path.join(build, "downloads");
const pounce = path.join(here, "pounce");
const POUNCE = ["pounce_solver-0.17.0-cp39-abi3-emscripten_4_0_9_wasm32.whl", "pyomo_pounce-0.17.0-py3-none-any.whl"];
const out = path.resolve(process.argv[2] ?? path.join(build, "leslie-gower.html"));

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
const FILES = {
  "pyodide.mjs": [PYODIDE, "635a6da3218fe4e5668da595acfe8b5ce77453d597d602f19a423dd250653441"],
  "pyodide.asm.js": [PYODIDE, "b22e5831eade9ff10e6fe2c811c68688cd91f10154377b4f80debcf5bafa1e56"],
  "pyodide.asm.wasm": [PYODIDE, "5effb6a1a6cc4a1a85bec4622701aa797c031e1de923cbbaf2ad47abdc4ab325"],
  "python_stdlib.zip": [PYODIDE, "71fee17f88a6260ec8c9c7c063533ee59c021fdc88a1ce76247378d3c4a35f4c"],
  "pyodide-lock.json": [PYODIDE, "f6e6f42f451f42affbbcddb00e8c9a3278dcbf399f57aab9f3f568839a7ff4a6"],
  "numpy-2.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl": [PYODIDE, "3db3c4f3e0448f4d62a85c262692f1260ccd8a91335442bd2442f21ffeddb558"],
  "pyomo-6.10.1-py3-none-any.whl": [
    "https://files.pythonhosted.org/packages/py3/p/pyomo/",
    "7a81b5301d4c0389809c1afd12311d23db04522341566332229d8ce4226bccb1",
  ],
};

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function download(name) {
  const [base, hash] = FILES[name];
  const file = path.join(downloads, name);
  if (existsSync(file) && sha256(readFileSync(file)) === hash) return file;
  console.log(`downloading ${base}${name}`);
  const response = await fetch(`${base}${name}`);
  if (!response.ok) throw new Error(`${base}${name}: ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (sha256(bytes) !== hash) throw new Error(`${name} does not have the SHA-256 ${hash}`);
  writeFileSync(file, bytes);
  return file;
}

mkdirSync(downloads, { recursive: true });
for (const name of Object.keys(FILES)) await download(name);

const read = (file) => readFileSync(file, "utf8").replaceAll("\r\n", "\n");
const code = (text) => text.replace(/^[ \t]*import .*;\n/gm, "").replace(/^export /gm, "");
const packed = (file) => gzipSync(readFileSync(file), { level: 9 }).toString("base64");

const worker = read(path.join(here, "worker.js"));

// The page's Python files by their path under /home/pyodide.
const python = { "sindae_page.py": read(path.join(here, "sindae_page.py")) };
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) walk(full);
    else if (name.endsWith(".py")) python[path.relative(repo, full).replaceAll("\\", "/")] = read(full);
  }
};
walk(path.join(repo, "sindae"));

const notices = `SiNDAE, by the Alves research group, github.com/Alves-research-group/SiNDAE, whose sindae package this file carries, and pounce 0.17.0's Python package for Pyodide and pyomo-pounce 0.17.0, whose wheels this file carries, are each under the Eclipse Public License 2.0, whose text follows. Pyodide 0.28.3, inside this file, is under the Mozilla Public License 2.0, the Python 3.13 standard library it carries under the Python Software Foundation License, NumPy 2.2.5, inside this file, under the BSD 3-Clause License, and Pyomo 6.10.1, inside this file, under the BSD 3-Clause License.

${read(path.join(repo, "LICENSE"))}`;

let page = read(path.join(here, "index.html"));
const put = (marker, text) => {
  if (!page.includes(marker)) throw new Error(`index.html has no ${marker}`);
  page = page.replace(marker, () => text);
};
for (const [name, text] of [["worker.js", worker], ["notices", notices], ...Object.entries(python)]) {
  if (text.includes("</script")) throw new Error(`${name} ends a script`);
}
put(">@notices<", `>${notices}<`);
put(">@worker.js<", `>${worker}<`);
const embedded = [...Object.keys(FILES).map((name) => [name, path.join(downloads, name)]), ...POUNCE.map((name) => [name, path.join(pounce, name)])];
put(">@pyodide<", `>globalThis.pyodideFiles = ${JSON.stringify(Object.fromEntries(embedded.map(([name, file]) => [name, packed(file)])))};<`);
put(">@python<", `>globalThis.pythonFiles = ${JSON.stringify(python)};<`);
put("// @worker.js\n", `${code(worker)}\n`);
mkdirSync(path.dirname(out), { recursive: true });
writeFileSync(out, page);
console.log(`${out} ${(page.length / 1e6).toFixed(1)} MB, ${Object.keys(python).length} Python files`);
