#!/usr/bin/env node
// barber CLI — trim an OpenAI-style message list from stdin.
//
//   npx barber-llm --keep 0.6 < messages.json > trimmed.json
//
// stdin:  a JSON array of {role, content} messages, or {"messages": [...]}.
// stdout: the trimmed messages as JSON.
// stderr: one summary line (chunks dropped, estimated tokens saved).

import { readFileSync } from "node:fs";
import { trim, VERSION } from "./index.js";

const USAGE = `usage: barber [--keep <0..1>] [--pretty] < messages.json > trimmed.json

Reads a JSON array of {role, content} messages (or {"messages": [...]}) from
stdin, drops query-irrelevant chunks, writes the trimmed messages to stdout.
Summary goes to stderr. Options:
  --keep <fraction>   fraction of chunks retained per block (default 0.6)
  --pretty            indent the output JSON
  --version           print version and exit`;

function fail(msg) {
  process.stderr.write(`barber: ${msg}\n`);
  process.exit(1);
}

let keep = null;
let pretty = false;
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "--help" || a === "-h") {
    console.log(USAGE);
    process.exit(0);
  } else if (a === "--version") {
    console.log(VERSION);
    process.exit(0);
  } else if (a === "--pretty") {
    pretty = true;
  } else if (a === "--keep") {
    keep = Number(argv[++i]);
  } else if (a.startsWith("--keep=")) {
    keep = Number(a.slice("--keep=".length));
  } else {
    fail(`unknown option ${a}\n${USAGE}`);
  }
}
if (keep !== null && !Number.isFinite(keep)) fail("--keep expects a number");

if (process.stdin.isTTY) {
  console.log(USAGE);
  process.exit(1);
}

let input;
try {
  input = JSON.parse(readFileSync(0, "utf8"));
} catch (e) {
  fail(`stdin is not valid JSON (${e.message})`);
}

let messages = input;
if (!Array.isArray(input)) {
  messages = input?.messages;
  if (keep === null && typeof input?.keep === "number") keep = input.keep;
}
if (!Array.isArray(messages)) {
  fail('expected a JSON array of messages or {"messages": [...]}');
}

const result = trim(messages, { keep: keep ?? 0.6 });
process.stdout.write(JSON.stringify(result.messages, null, pretty ? 2 : 0) + "\n");
process.stderr.write(
  `barber: dropped ${result.chunksDropped} chunk(s), ~${result.tokensSaved} tokens saved\n`
);
