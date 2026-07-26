import test from "node:test";
import assert from "node:assert/strict";
import { Cache, makeTransform, trim } from "../index.js";
import { pyRound, charLen } from "../core.js";

test("Cache evicts oldest beyond maxsize", () => {
  const c = new Cache({ maxsize: 2 });
  c.set("a", ["A", true]);
  c.set("b", ["B", true]);
  c.set("c", ["C", true]);
  assert.equal(c.size, 2);
  assert.equal(c.has("a"), false);
  assert.deepEqual(c.get("c"), ["C", true]);
});

test("Cache membership hit refreshes recency", () => {
  const c = new Cache({ maxsize: 2 });
  c.set("a", ["A", true]);
  c.set("b", ["B", true]);
  assert.equal(c.has("a"), true); // refresh: b is now oldest
  c.set("c", ["C", true]);
  assert.equal(c.has("a"), true);
  assert.equal(c.has("b"), false);
});

// A block, once decided, is replayed verbatim on later turns even when the
// question changes — that is the whole point of keying on content only.
test("freeze-on-first-sight across turns with a shared cache", () => {
  const chunkA = "Apples keep best in the cold drawer and last for weeks when dry, the orchard guide says so in its storage chapter for autumn varieties and notes the cellar option too.";
  const chunkB = "Bolts for the deck frame are galvanized half inch lag screws, torqued snug then a quarter turn, per the framing sheet from the hardware supplier in town.";
  const filler1 = "The community workshop calendar lists a cider pressing weekend in October with sign-up sheets posted by the door and a loaner press for members who bring clean containers.";
  const filler2 = "Garden plot renewals open in January, returning members keep their plot numbers and new members draw lots for the beds along the south fence near the water taps.";
  const filler3 = "The tool library added a laser level and two more dehydrators to the catalog this spring after the fundraiser cleared its goal at the pancake breakfast.";
  const block = [chunkA, filler1, chunkB, filler2, filler3].join("\n\n");

  const q1 = [{ role: "user", content: block }, { role: "user", content: "How should apples be stored?" }];
  const q2 = [{ role: "user", content: block }, { role: "user", content: "Which bolts hold the deck frame?" }];

  // Fresh caches: different questions select differently (the test has teeth).
  const fresh1 = trim(structuredClone(q1), { keep: 0.4 });
  const fresh2 = trim(structuredClone(q2), { keep: 0.4 });
  assert.notEqual(fresh1.messages[0].content, fresh2.messages[0].content);

  // Shared cache: the second question replays the first decision verbatim.
  const cache = new Cache();
  const turn1 = trim(structuredClone(q1), { keep: 0.4, cache });
  const turn2 = trim(structuredClone(q2), { keep: 0.4, cache });
  assert.equal(turn1.messages[0].content, turn2.messages[0].content);
});

test("makeTransform returns the pipeline pair and reports stats", () => {
  const [name, fn] = makeTransform({ keep: 0.5 });
  assert.equal(name, "barber");
  const block = Array.from({ length: 6 }, (_, i) =>
    `Paragraph ${i} of the maintenance digest covers a distinct subsystem in enough words to pass the size gate for selection when joined with its five siblings into one message block.`
  ).join("\n\n");
  const [out, changed] = fn([
    { role: "user", content: block },
    { role: "user", content: "What does paragraph two cover in the digest?" },
  ]);
  assert.equal(out.length, 2);
  assert.equal(typeof changed, "boolean");
  assert.ok(fn.lastStats.chunksIn === 0 || fn.lastStats.chunksIn >= fn.lastStats.chunksKept);
});

test("pyRound matches Python round() half-to-even", () => {
  assert.equal(pyRound(2.5), 2);
  assert.equal(pyRound(3.5), 4);
  assert.equal(pyRound(2.4), 2);
  assert.equal(pyRound(2.6), 3);
  assert.equal(pyRound(0.5), 0);
});

test("charLen counts code points, not UTF-16 units", () => {
  assert.equal(charLen("😀🚀"), 2);
  assert.equal("😀🚀".length, 4);
  assert.equal(charLen("plain"), 5);
});

test("async embedders are rejected with a clear error", () => {
  const block = Array.from({ length: 5 }, (_, i) =>
    `Section ${i} of the operations handbook describes procedures at a length sufficient to qualify the whole message for selection under the default configuration thresholds.`
  ).join("\n\n");
  assert.throws(
    () =>
      trim(
        [
          { role: "user", content: block },
          { role: "user", content: "What does section three describe?" },
        ],
        { embedder: async (texts) => texts.map(() => [1, 0]) }
      ),
    /async embedders/
  );
});
