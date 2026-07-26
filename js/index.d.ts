// Type definitions for barber-llm.
// Field names are the camelCase forms of the Python package's snake_case
// API (min_message_chars -> minMessageChars, tokens_saved -> tokensSaved).

export interface ContentPart {
  type: string;
  text?: string;
  [key: string]: unknown;
}

export interface Message {
  role: string;
  content?: string | ContentPart[] | unknown;
  [key: string]: unknown;
}

export interface SelectionConfig {
  /** Only consider a message for selection at or above this many chars. */
  minMessageChars: number;
  /** Only split/select when the message has at least this many chunks. */
  minChunks: number;
  minKeepRatio: number;
  maxKeepRatio: number;
  /** Chunks scoring below top*relativeFloor are drop candidates. */
  relativeFloor: number;
  /** Always keep the first & last chunk of a block. */
  keepLeadTail: boolean;
  /** Marker inserted where a run of chunks was dropped; {n} is the count. */
  dropMarker: string;
  /** Roles whose large messages are selectable. */
  selectableRoles: readonly string[];
}

/** Term-weight bag (Map) or dense vector (array / typed array). */
export type Vector = Map<string, number> | number[] | Float32Array | Float64Array;

/** Synchronous embedder: texts in, one vector per text out. */
export type Embedder = (texts: string[]) => Vector[];

export interface TrimResult {
  /** Trimmed messages; originals are not mutated. */
  messages: Message[];
  /**
   * Estimated tokens saved (code points / 4). Signed: negative means the
   * drop markers cost more than the dropped chunks saved.
   */
  tokensSaved: number;
  chunksDropped: number;
  changed: boolean;
}

export interface CacheLike {
  has(key: string): boolean;
  get(key: string): [string, boolean] | undefined;
  set(key: string, value: [string, boolean]): unknown;
}

export interface TrimOptions {
  /** Fraction of chunks retained per block. Benchmark default: 0.6. */
  keep?: number;
  /** Custom embedder; defaults to the zero-dependency lexical scorer. */
  embedder?: Embedder | null;
  cfg?: Partial<SelectionConfig> | null;
  /** Shared decision cache for freeze-on-first-sight across turns. */
  cache?: CacheLike | Map<string, [string, boolean]> | null;
}

export type Transform = ((messages: Message[]) => [Message[], boolean]) & {
  lastStats: {
    blocksProcessed: number;
    chunksIn: number;
    chunksKept: number;
    charsIn: number;
    charsOut: number;
  };
};

/** Bounded LRU decision cache for long-running processes. */
export class Cache implements CacheLike {
  constructor(options?: { maxsize?: number });
  maxsize: number;
  has(key: string): boolean;
  get(key: string): [string, boolean] | undefined;
  set(key: string, value: [string, boolean]): this;
  readonly size: number;
  keys(): string[];
  values(): [string, boolean][];
  [Symbol.iterator](): IterableIterator<[string, [string, boolean]]>;
}

export function trim(messages: Message[], options?: TrimOptions): TrimResult;

export function makeTransform(options?: TrimOptions): ["barber", Transform];

export function makeSelectionTransform(options?: {
  embedFn?: Embedder | null;
  tier?: "simple" | "mid" | "medium" | "complex" | "reasoning" | string;
  complexityScore?: number | null;
  cfg?: Partial<SelectionConfig> | null;
  decisionCache?: CacheLike | Map<string, [string, boolean]> | null;
}): ["context_selection", Transform];

/** Zero-dependency deterministic lexical (BM25-lite) embedder factory. */
export function lexical(): Embedder;

export function splitChunks(text: string): string[];
export function textOf(content: unknown): string;

export const DEFAULT_CONFIG: Readonly<SelectionConfig>;
export const TIER_KEEP_RATIO: Readonly<Record<string, number>>;
export const VERSION: string;
