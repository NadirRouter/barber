import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const CLI = fileURLToPath(new URL("../cli.js", import.meta.url));
const golden = JSON.parse(
  readFileSync(new URL("./fixtures/golden.json", import.meta.url), "utf8")
);

const run = (args, input) =>
  spawnSync(process.execPath, [CLI, ...args], { input, encoding: "utf8" });

test("cli trims a message array from stdin", () => {
  const c = golden.cases.find((x) => x.name === "rag_basic");
  const r = run(["--keep", String(c.keep)], JSON.stringify(c.messages));
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(JSON.parse(r.stdout), c.expected.messages);
  assert.match(r.stderr, /dropped \d+ chunk/);
});

test("cli accepts {messages, keep} wrapper", () => {
  const c = golden.cases.find((x) => x.name === "bankers_rounding");
  const r = run([], JSON.stringify({ messages: c.messages, keep: c.keep }));
  assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(JSON.parse(r.stdout), c.expected.messages);
});

test("cli rejects invalid JSON", () => {
  const r = run([], "not json");
  assert.equal(r.status, 1);
  assert.match(r.stderr, /not valid JSON/);
});

test("cli rejects non-array payloads", () => {
  const r = run([], JSON.stringify({ nope: true }));
  assert.equal(r.status, 1);
  assert.match(r.stderr, /expected a JSON array/);
});

test("cli --version prints the version", () => {
  const r = run(["--version"], "");
  assert.equal(r.status, 0);
  assert.match(r.stdout, /^\d+\.\d+\.\d+\n$/);
});
