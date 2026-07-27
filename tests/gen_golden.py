#!/usr/bin/env python3
"""Generate golden parity fixtures for the JS port (js/test/fixtures/golden.json).

Runs the Python implementation (the source of truth) on fixed scenarios and
records the expected TrimResult for each. The JS test suite replays these and
must match exactly — messages byte-for-byte, counts number-for-number.

Regenerate after any (re-benchmarked, upstream-approved) change to the
selection logic:   python tests/gen_golden.py
CI freshness gate: python tests/gen_golden.py --check

The token counter is pinned to the dependency-free fallback (len // 4)
because the JS package always uses it; tiktoken presence must not change
the fixtures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import barber  # noqa: E402
from barber import SelectionConfig, trim  # noqa: E402

barber._token_counter = lambda: (lambda s: len(s) // 4)

OUT_PATH = ROOT / "js" / "test" / "fixtures" / "golden.json"

_CFG_KEYS = {
    "minMessageChars": "min_message_chars",
    "minChunks": "min_chunks",
    "minKeepRatio": "min_keep_ratio",
    "maxKeepRatio": "max_keep_ratio",
    "relativeFloor": "relative_floor",
    "keepLeadTail": "keep_lead_tail",
    "dropMarker": "drop_marker",
}


def _cfg(camel: dict) -> SelectionConfig:
    return SelectionConfig(**{_CFG_KEYS[k]: v for k, v in camel.items()})


def block(*chunks: str) -> str:
    return "\n\n".join(chunks)


def utf16_units(s: str) -> int:
    """What JS String.prototype.length would report for this string."""
    return len(s.encode("utf-16-le")) // 2


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

S1_CTX = block(
    "Meridian is a distributed job scheduler for data teams. This page collects operational notes gathered from production deployments across three release lines and the managed cloud.",
    "When a job fails, Meridian applies exponential retry backoff: the first retry waits ten seconds, and each further attempt doubles the delay up to a ceiling of fifteen minutes per queue.",
    "The dashboard ships with three color themes. Theme selection lives under profile preferences and applies per browser, not per account, so each workstation remembers its own choice.",
    "Billing is metered per executed task minute. Invoices are issued on the first business day of the month and unused committed minutes roll over for one quarter before expiring.",
    "Retry budgets are configurable per queue: a job that exhausts its retry allowance is parked in the dead letter queue and Meridian stops scheduling further attempts for it.",
    "Installation on bare metal uses the meridian-agent package from our apt and yum repositories. Containerized installs pull the signed image from the public registry mirror.",
    "The company logo went through four revisions before the current compass rose design. Early drafts featured a stylized crane, retired after a round of user testing.",
    "A failed job records its final error, the attempt count, and the last worker id. These fields surface in the failures view and export cleanly to CSV for postmortems.",
    "Keyboard shortcuts: press g then q to jump to queues, g then w for workers, and question mark to open the shortcut overlay from any screen in the dashboard.",
    "Support hours are Monday through Friday, nine to five UTC. Enterprise plans add a paged on-call rotation with a fifteen minute response target for incidents.",
)

S2_CTX = block(
    "Reports overview: Corvid Analytics generates weekly summaries, ad hoc snapshots, and scheduled exports for every workspace on the growth plan and above.",
    "Scheduled report exports are capped at two hundred megabytes per file; larger result sets are split into numbered parts and delivered together as a single archive.",
    "All customer records must be encrypted at rest, and access to raw exports is limited to workspace administrators.",
    "Rotate the api_key used by the export webhook every ninety days; rotation happens in the integrations panel and takes effect within one minute.",
    "Chart colors follow the workspace palette. Custom palettes accept up to twelve hex values and fall back to the default ramp when a series runs past the last color.",
    "The mobile app renders dashboards read-only. Editing layouts, changing owners, and archiving boards remain desktop actions in this release cycle.",
    "Export delivery destinations include email, object storage buckets, and the webhook relay; deliveries that bounce are retried three times before alerting the owner.",
    "Historical note: the exporter was rewritten in 2024 to stream rows instead of buffering whole result sets in memory, cutting peak usage by an order of magnitude.",
)

S3_CTX = block(
    "Logistics digest for the third week of March, covering inbound freight, carrier performance, and warehouse throughput across the Rotterdam and Halifax lanes.",
    "Carrier on-time performance dipped to ninety one percent, driven by winter storms along the northern corridor and a dockworker slowdown at the transfer hub.",
    "Warehouse throughput held steady at four thousand pallets per day, with pick accuracy at ninety nine point six percent after the scanner firmware update.",
    "Insurance premiums for high value freight renew in April; the broker projects a low single digit increase given the year's clean loss history.",
    "The zephyrite shipment sat in a customs inspection queue for six days because the mineral classification code on the manifest disagreed with the code in the import permit.",
    "Fuel surcharges tick down two percent next quarter under the new carrier agreement, applied automatically to lanes billed on the standard rate card.",
    "Next digest lands in two weeks; send corrections to the logistics channel before Thursday to make the editorial cutoff for the print edition.",
)

S4_CHUNKS = [
    "Assembly line notes for the model twelve conveyor, collected by the maintenance crew during the spring service window and shared with the floor leads ahead of the summer production push.",
    "Sprocket alignment tolerance is half a millimeter measured at the drive shaft; re-check the sprocket alignment after any belt swap because tolerance drift compounds over a full shift.",
    "During sprocket alignment the crew locks the drive, marks the datum edge, and torques the carrier bolts in a star pattern before measuring runout with the dial indicator.",
    "A worn sprocket shows hooked teeth under the inspection lamp; swap it in pairs with its chain so the sprocket surfaces wear evenly across the whole assembly.",
    "General housekeeping: return torque wrenches to the shadow board and log consumables in the bin ledger before the end of each shift so the next crew starts from a clean count.",
]
S4_CTX = block(*S4_CHUNKS)

S5_CHUNKS = [
    "Data pipeline runbook, stage by stage, maintained by the platform group; read the overview before touching production schedules or backfill windows, and record every manual action in the log.",
    "Stage one ingests raw events from the collector fleet, validates schemas against the registry, and lands parquet files partitioned by hour into the bronze data bucket for downstream consumers.",
    "Trivia: our first prototype ran on someone's spare laptop under a desk.",
    "Stage two compacts small files, deduplicates on event id, and publishes silver tables consumed by the metrics layer and the feature store refresh that runs every morning before standup.",
    "Escalation path: page the platform rotation for stage failures older than one hour, and file a ticket for anything recoverable by a scheduled backfill in the next maintenance window tonight.",
]
S5_CTX = block(*S5_CHUNKS)

S6_TOOL = block(
    "Migration plan for the catalog service, drafted by the data platform team and reviewed in the March architecture sync before the change board sign-off.",
    "Step one snapshots the primary database and copies the archive to cold storage, tagging the snapshot with the release identifier for later verification.",
    "Step two applies the schema changes behind a feature gate, creating the new columns and backfilling defaults in batches of fifty thousand rows.",
    "Step three rebuilds the search index from the migrated tables, streaming documents through the analyzer fleet and swapping the alias once the new index reports healthy.",
    "Step four migrates user preferences to the new keyspace, preserving timestamps so recently-active sessions keep their unsaved layout choices.",
    "Step five re-registers plugins against the new plugin manifest and disables any plugin that fails its handshake twice in a row.",
    "Step six warms the edge caches by replaying the top ten thousand queries from the previous week at a controlled request rate.",
    "Step seven wires the new dashboards and alert rules, then routes the on-call runbook links to the refreshed monitoring pages.",
    "Step eight documents the rollback procedure: restore the snapshot, re-point the alias to the previous search index, and revert the feature gate.",
)

S6_FUNC = block(
    "Rollout checklist returned by the release tracker for build 2214, current as of this morning's sync and subject to the change freeze calendar.",
    "Feature flags: confirm the catalog gate is off in production and on in staging, with ownership recorded in the flag registry.",
    "Canary: route five percent of traffic to the new build for one hour and compare error rates against the control cohort before widening.",
    "Search verification: after the index rebuild completes, run the golden query set and diff the top ten results against the recorded baseline.",
    "Comms: post the rollout window in the release channel and update the status page banner if customer-visible latency is expected.",
    "Dashboards: pin the migration board to the on-call view for the duration of the rollout and unpin it at the retro.",
    "Retro: schedule the review within one week of completion and attach the timing data from the tracker to the agenda.",
)

S6_ASSISTANT = (
    "Analysis: the migration plan is sequenced so the search index rebuild lands before user preference migration, which matters because the "
    "preference keyspace references document ids that only exist once the new index is live. The canary hour gives the analyzer fleet time to "
    "surface tokenization regressions before the alias swap, and the golden query diff is the only step that exercises ranking end to end, so "
    "treat a ranking drift there as a blocker rather than a warning. Rollback stays cheap until step four; after preferences migrate, restoring "
    "the snapshot loses a day of layout edits, so the decision point for aborting is the end of step three. Cache warmup can overlap the "
    "verification window safely since it only issues read traffic at a controlled rate against the new alias."
)

S7_PARTS_TEXT = (
    "Longform note carried as a content part: the quarterly planning template asks each team to list capacity, committed work, stretch goals, "
    "and known risks, then walks through dependency mapping with the partner teams before locking the plan. The template's final section "
    "collects the metrics each team will watch weekly, the thresholds that trigger a replan, and the single named owner for every committed "
    "line item. Historically the teams that fill in the risk section honestly replan half as often, because the dependency map catches the "
    "cross-team collisions while the quarter is still young and the mitigation is still a one-day conversation instead of a two-week scramble. "
    "The planning doc closes with a short retro template for the end of the quarter, four questions long, kept deliberately identical across "
    "teams so the aggregate rollup stays comparable from one planning cycle to the next without any manual reformatting effort."
)

S7_THREE_CHUNK = block(
    "First section of the archived onboarding guide, covering account provisioning from the request form through the identity provider group "
    "assignments, including the approval chain for elevated roles and the audit trail every approval writes to the compliance ledger for later "
    "review by the quarterly access certification campaign, which samples one in five grants and walks the chain backwards to the original request.",
    "Second section, covering workstation setup: the managed laptop image, the update rings and their deferral windows, the self-service "
    "application catalog, and the escalation path when a build tool needs an exception to the standard image, which goes through the platform "
    "engineering intake board and comes back with either an approved exception or a supported alternative inside five business days.",
    "Third section, covering the first-week checklist: meet the onboarding buddy, complete the compliance modules, ship a starter change through "
    "the full review pipeline, and book the thirty-day check-in with the hiring manager, whose template lives next to this guide and gets "
    "updated every quarter from the feedback the new joiners leave in the anonymous onboarding survey form.",
)

S7_ASSISTANT = block(
    "Assistant-authored summary block one: the support backlog splits into password resets, billing disputes, and integration questions, with "
    "integration questions taking triple the handle time of the other two categories combined across every quarter measured so far.",
    "Assistant-authored summary block two: deflection from the knowledge base rose after the search upgrade, and the largest single win came "
    "from surfacing the webhook troubleshooting article on the integrations landing page where the ticket form used to be.",
    "Assistant-authored summary block three: the billing dispute queue clears fastest on Tuesdays when the finance liaison joins the rotation, "
    "suggesting the bottleneck is authority to issue credits rather than investigation time in the support tool itself.",
    "Assistant-authored summary block four: proposed next quarter targets are a ten percent reduction in first response time and a knowledge "
    "base article for each of the top twenty integration errors, owned jointly by support and the platform documentation guild.",
    "Assistant-authored summary block five: the on-call shadowing program graduated four new responders this quarter and the post-shift survey "
    "scores the pairing sessions as the single most useful part of the ramp, ahead of the runbook reading list.",
    "Assistant-authored summary block six: tooling asks from the team are a canned-response linter, a duplicate-ticket detector, and a way to "
    "preview the customer's plan limits inline instead of opening the admin panel in a second tab for every conversation.",
)

S7_FINAL_USER = block(
    "Here is the situation I need help with this morning, written out in full so nothing gets lost between threads and hand-offs from the "
    "overnight rotation, whose notes I have pasted below verbatim where relevant to the open questions.",
    "A customer on the enterprise plan reports that their scheduled export arrived twice on Monday and not at all on Tuesday, and their "
    "webhook receiver logged one delivery attempt each day with different payload sizes on each attempt.",
    "Their integration uses the standard relay with retries enabled, and the workspace audit log shows a manual export triggered by an "
    "automation user at roughly the same minute as the scheduled run on both days in question.",
    "The account team wants a summary they can forward, the support engineer wants to know whether to disable the automation user, and the "
    "customer mainly wants to stop receiving duplicate files in their bucket every Monday morning.",
    "Given all of that, what sequence of checks would you run first, and which of the two exports should be turned off while the "
    "investigation is open so the customer keeps exactly one reliable delivery per day?",
)

EMOJI = "😀🚀🎉🔥"  # four astral code points, two UTF-16 units each

S8_TRAP = block(
    "Launch party planning notes with reactions from the crew " + EMOJI * 24,
    "Venue shortlist reactions collected from the poll thread " + EMOJI * 24,
    "Catering options ranked by the tasting committee last week " + EMOJI * 24,
    "Playlist suggestions gathered from the team channel today " + EMOJI * 24,
)

S8_REAL = block(
    "Beta program overview for the spring cohort 😀 covering enrollment, device coverage, and the feedback loop the product crew runs each week with the coordinators.",
    "The beta launch is scheduled for April 9 🚀 with invitations going out in three waves so the support rotation can absorb the onboarding questions without paging anyone.",
    "Wave one covers internal dogfooders and the customer council 🎉 wave two adds the waitlist accounts, and wave three opens self-serve enrollment from the site banner.",
    "Feedback lands in the beta board, triaged every morning; crash reports page the build cop directly 🔥 and everything else waits for the weekly review with the leads.",
    "Swag inventory note: the sticker sheets from the winter batch are nearly gone and the reorder decision belongs to the community team, not the beta coordinators.",
    "Exit criteria for the beta are two weeks below the crash budget, survey satisfaction above four out of five, and the migration tool passing its dry run on the oldest cohort.",
)

S9_CTX = block(
    "Solar inverter warranty guide for residential installations, covering coverage terms, the claim process, registration, and transfer rules for the current product line.",
    "Facilities note: the company picnic moved to the lakeside pavilion this year and the shuttle leaves the north lot every twenty minutes starting at ten.",
    "The standard inverter warranty runs ten years from the installation date, and registering the serial number within ninety days extends coverage to twelve years at no cost.",
    "Reminder from facilities: the office recycling program now accepts soft plastics in the green bins next to the loading dock on the ground floor.",
    "Warranty claims start in the installer portal: submit the serial number, the fault code from the display, and a photo of the nameplate, then book the inspection slot.",
    "The parking garage resurfacing continues through the end of the month; levels two and three close on alternating weekends per the posted schedule.",
    "An approved claim ships a replacement inverter within five business days, and the courier collects the faulty unit at delivery using the same packaging.",
    "The cafeteria menu rotates to the summer cycle next week, headlined by the return of the grain bowl station and the cold brew tap on the mezzanine.",
    "Warranty transfer to a new homeowner is automatic when the property sells; the new owner re-registers the serial number to keep the extended term active.",
    "The book club meets Thursday in the small library to discuss the first half of the maintenance manual novelization, snacks provided by last month's host.",
    "Damage from unapproved third-party installers or from grid surges outside the rated envelope falls outside the warranty, and the portal flags those fault codes at intake.",
    "This guide is maintained by the field service team; send corrections through the documentation channel and expect the next revision at the quarterly refresh.",
)

S10_BLOCK = block(
    "Plan comparison notes assembled for the sales enablement wiki, refreshed after the spring packaging change and the new usage meters went live.",
    "The premium plan includes unlimited projects, five hundred automation runs per month, priority routing for support tickets, and the audit log with one year of retention.",
    "The starter plan covers three projects and fifty automation runs, with community support and a thirty day activity view instead of the full audit log.",
    "Annual billing applies a fifteen percent discount to either plan and locks the seat price for the term; monthly billing floats with the published rate card.",
    "Trials run fourteen days on premium features with no card required, and a workspace downgrades to starter automatically when the trial ends without a purchase.",
    "Data export is available on both plans from the workspace settings page, and closed accounts keep export access for sixty days after cancellation takes effect.",
)

S11_CTX = block(
    "The old town walking route begins at the clock tower and follows the river wall past the merchant houses toward the stone bridge and its twin gatehouses.",
    "Local bakeries open before sunrise, and the corner shop by the fountain sells the seeded loaf that the market guides mention in every seasonal pamphlet.",
    "The maritime museum keeps the tide tables from the harbor's founding century on display beside the restored pilot boat and the original lens from the lighthouse.",
    "Tram line four circles the hill district and offers the best rooftop views on the descent toward the botanical garden and the glasshouse promenade.",
    "The evening market on the quay runs Thursday through Sunday, with the fish stalls closing first and the spice sellers staying open until the last ferry.",
    "A city pass covers the funicular, both museums, and the bridge towers, and it pays for itself within a single afternoon of ordinary sightseeing.",
)

S12_CTX = block(
    "Weekly minerals desk notes for subscribers, spanning the metals board and the specialty stones.",
    "Copper held flat through the week while the exchange warehouses reported thinner inventories.",
    "Garnet price today sits at forty two dollars per carat for gem grade at the Jaipur auction.",
    "The garnet price climbed four percent this week on jewelry demand ahead of the festival season.",
    "Dealers expect the garnet market to cool once the festival orders clear the cutting houses.",
    "Next week's note adds the lithium board and a spotlight on the new assay rules for exporters.",
)

# --- agent-shaped scenarios (line-oriented tool output, content parts) -------

def read_output(*lines: str) -> str:
    """What a Read tool returns: `N\t` line prefixes, so the blank lines of the
    underlying file are no longer blank."""
    return "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(lines))


_SRC = []
for _i in range(16):
    _SRC += [
        f"def unrelated_helper_number_{_i}(argument, other_argument):",
        f"    intermediate = compute_something_unrelated_{_i}(argument)",
        "    return intermediate + other_argument",
        "",
    ]
_SRC += ["def parse_yaml_config(path):", "    return yaml.safe_load(path)"]
S_READ = read_output(*_SRC)

S_GREP = "\n".join(
    f"src/module_{i}/handler.py:{i * 3}: def handle_event_{i}(payload, context):"
    for i in range(60)
)

S_DIFF = "\n".join(
    ["diff --git a/svc.py b/svc.py", "--- a/svc.py", "+++ b/svc.py"]
    + [
        line
        for i in range(6)
        for line in (
            f"@@ -{i * 20 + 1},4 +{i * 20 + 1},4 @@ def region_{i}():",
            f"-    old_value_{i} = compute_legacy_{i}()",
            f"+    new_value_{i} = compute_modern_{i}()",
            f"     unchanged_line_{i} = passthrough_{i}()",
        )
    ]
)

S_JSON = (
    "{\n"
    + "\n".join(f'  "setting_number_{i}": "configured value {i}",' for i in range(40))
    + '\n  "final_setting": true\n}'
)

SCENARIOS = [
    {
        "name": "rag_basic",
        "keep": 0.6,
        "wantChanged": True,
        "messages": [
            {"role": "system", "content": "You are a concise support assistant for Meridian."},
            {"role": "user", "content": S1_CTX},
            {"role": "user", "content": "How does Meridian handle retry backoff for failed jobs?"},
        ],
    },
    {
        "name": "pin_deontic_and_pii",
        "keep": 0.45,
        "wantChanged": True,
        "messages": [
            {"role": "system", "content": "Answer from the provided context only."},
            {"role": "tool", "content": S2_CTX},
            {"role": "user", "content": "What is the export size limit for scheduled reports?"},
        ],
    },
    {
        "name": "rare_entity_multihop",
        "keep": 0.45,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S3_CTX},
            {"role": "user", "content": "Why did the zephyrite shipment clear customs late?"},
        ],
    },
    {
        "name": "bankers_rounding",
        "keep": 0.5,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S4_CTX},
            {"role": "user", "content": "What is the sprocket alignment tolerance on the drive shaft?"},
        ],
    },
    {
        "name": "negative_savings",
        "keep": 0.8,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S5_CTX},
            {"role": "user", "content": "What happens in stage two of the data pipeline?"},
        ],
    },
    {
        "name": "multi_block_roles",
        "keep": 0.6,
        "wantChanged": True,
        "messages": [
            {"role": "system", "content": "You are the release engineering copilot."},
            {"role": "tool", "content": S6_TOOL},
            {"role": "assistant", "content": S6_ASSISTANT},
            {"role": "function", "content": S6_FUNC},
            {"role": "user", "content": "Which migration step rebuilds the search index?"},
        ],
    },
    {
        "name": "noop_gates",
        "keep": 0.6,
        "wantChanged": False,
        "messages": [
            {"role": "system", "content": "You are a support assistant for the Acme knowledge base."},
            {"role": "user", "content": "Quick note: the export ran fine yesterday, context attached in the next messages for the real question coming later."},
            {"role": "user", "content": [{"type": "text", "text": S7_PARTS_TEXT}]},
            {"role": "user", "content": S7_THREE_CHUNK},
            {"role": "assistant", "content": S7_ASSISTANT},
            {"role": "user", "content": S7_FINAL_USER},
        ],
    },
    {
        "name": "emoji_length_gates",
        "keep": 0.6,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S8_TRAP},
            {"role": "user", "content": S8_REAL},
            {"role": "user", "content": "When is the beta launch scheduled and who is in wave one?"},
        ],
    },
    {
        "name": "scattered_markers",
        "keep": 0.5,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S9_CTX},
            {"role": "user", "content": "How long is the solar inverter warranty and how do I claim it?"},
        ],
    },
    {
        "name": "duplicate_block_cache",
        "keep": 0.6,
        "wantChanged": True,
        "messages": [
            {"role": "system", "content": "Answer plan questions from the notes."},
            {"role": "tool", "content": S10_BLOCK},
            {"role": "assistant", "content": "Fetched the plan comparison notes twice to double-check freshness."},
            {"role": "tool", "content": S10_BLOCK},
            {"role": "user", "content": "What does the premium plan include?"},
        ],
    },
    {
        "name": "zero_overlap_collapse",
        "keep": 1.0,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S11_CTX},
            {"role": "user", "content": "מהי מדיניות ההחזרות בחנות המוזיאון?"},
        ],
    },
    {
        "name": "custom_cfg_marker",
        "keep": 0.5,
        "wantChanged": True,
        "cfg": {"dropMarker": "[snip {n} |*]", "minMessageChars": 200, "keepLeadTail": False},
        "messages": [
            {"role": "user", "content": S12_CTX},
            {"role": "user", "content": "What is the garnet price today?"},
        ],
    },
    {
        "name": "agent_read_content_parts",
        "keep": 0.6,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": "read the config module"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"file_path": "/m.py"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": S_READ}
                ],
            },
            {"role": "user", "content": "what does parse_yaml_config do?"},
        ],
    },
    {
        "name": "agent_grep_windows",
        "keep": 0.5,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S_GREP},
            {"role": "user", "content": "which handler deals with module_7?"},
        ],
    },
    {
        "name": "agent_diff_hunks",
        "keep": 0.5,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": S_DIFF},
            {"role": "user", "content": "what changed in region_3?"},
        ],
    },
    {
        "name": "agent_json_never_line_chunked",
        "keep": 0.5,
        "wantChanged": False,
        "messages": [
            {"role": "user", "content": S_JSON},
            {"role": "user", "content": "what is setting_number_12 set to?"},
        ],
    },
    {
        # The newest tool result is the one the agent is about to act on, and it
        # sits in the last user message — the "never prune the question" guard
        # covers it. Selection reaches it on the next turn, not this one.
        "name": "agent_newest_toolresult_is_untouched",
        "keep": 0.6,
        "wantChanged": False,
        "messages": [
            {"role": "user", "content": "what does parse_yaml_config do?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"file_path": "/m.py"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": S_READ}
                ],
            },
        ],
    },
    {
        # A full agent loop: the question, a big tool result, then another tool
        # call. The query has to be found by walking back past tool-result
        # messages that carry no text of their own.
        "name": "agent_query_walks_back_past_toolresults",
        "keep": 0.6,
        "wantChanged": True,
        "messages": [
            {"role": "user", "content": "what does parse_yaml_config do?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read",
                     "input": {"file_path": "/m.py"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": S_READ}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "Bash",
                     "input": {"command": "pytest -q"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "1 passed"}
                ],
            },
        ],
    },
]


def build() -> dict:
    cases = []
    for sc in SCENARIOS:
        kwargs = {"keep": sc["keep"]}
        if sc.get("cfg"):
            kwargs["cfg"] = _cfg(sc["cfg"])
        r = trim(sc["messages"], **kwargs)
        assert r.changed == sc["wantChanged"], f"{sc['name']}: changed={r.changed}"
        cases.append(
            {
                "name": sc["name"],
                "keep": sc["keep"],
                "cfg": sc.get("cfg"),
                "messages": sc["messages"],
                "expected": {
                    "messages": r.messages,
                    "tokensSaved": r.tokens_saved,
                    "chunksDropped": r.chunks_dropped,
                    "changed": r.changed,
                },
            }
        )
    return {
        "generator": "tests/gen_golden.py",
        "barberVersion": barber.__version__,
        "tokenCounter": "len // 4 (code points)",
        "cases": cases,
    }


def sanity(cases: list) -> None:
    """Lock the traps each scenario exists for, so fixture drift is loud."""
    by = {c["name"]: c for c in cases}

    # bankers_rounding: round(5 * 0.5) must be 2 (half-to-even), so chunk 3
    # is dropped; a half-up port keeps all five and changes nothing.
    b = by["bankers_rounding"]["expected"]["messages"][0]["content"]
    assert S4_CHUNKS[3] not in b and S4_CHUNKS[1] in b and S4_CHUNKS[2] in b

    # negative_savings: the marker costs more than the dropped chunk saved.
    assert by["negative_savings"]["expected"]["tokensSaved"] < 0
    real5 = by["negative_savings"]["expected"]["messages"][0]["content"]
    assert S5_CHUNKS[2] not in real5

    # emoji_length_gates: the trap block is >=800 UTF-16 units but <800 code
    # points, so Python skips it; a .length port would trim it.
    assert len(S8_TRAP) < 800 <= utf16_units(S8_TRAP), (len(S8_TRAP), utf16_units(S8_TRAP))
    trap_out = by["emoji_length_gates"]["expected"]["messages"][0]["content"]
    assert trap_out == S8_TRAP
    assert by["emoji_length_gates"]["expected"]["messages"][1]["content"] != S8_REAL

    # duplicate_block_cache: the second identical block replays the first
    # decision byte-for-byte.
    dup = by["duplicate_block_cache"]["expected"]["messages"]
    assert dup[1]["content"] == dup[3]["content"] != S10_BLOCK

    # zero_overlap_collapse: no shared token at keep=1.0 collapses to
    # lead + tail with one four-chunk marker (known locked behavior).
    col = by["zero_overlap_collapse"]["expected"]["messages"][0]["content"]
    assert "4 passage(s) omitted" in col

    # custom_cfg_marker: leading AND trailing markers, custom template with
    # regex-special characters, keepLeadTail off (lead/tail dropped).
    cm = by["custom_cfg_marker"]["expected"]["messages"][0]["content"]
    assert cm.startswith("[snip 2 |*]") and cm.endswith("[snip 2 |*]")
    assert by["custom_cfg_marker"]["expected"]["chunksDropped"] == 4

    # every selectable block in changed scenarios really is >= 800 chars
    for name in ("rag_basic", "pin_deontic_and_pii", "rare_entity_multihop",
                 "bankers_rounding", "negative_savings", "scattered_markers"):
        blk = by[name]["messages"][0 if by[name]["messages"][0]["role"] != "system" else 1]
        assert len(blk["content"]) >= 800, name

    # noop_gates: untouched means the exact same message objects came back
    noop = by["noop_gates"]
    assert noop["expected"]["messages"] == noop["messages"]
    assert noop["expected"]["tokensSaved"] == 0 and noop["expected"]["chunksDropped"] == 0


def dumps(data: dict) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed fixtures match a fresh run")
    args = ap.parse_args()

    data = build()
    sanity(data["cases"])
    text = dumps(data)

    if args.check:
        if not OUT_PATH.exists() or OUT_PATH.read_text() != text:
            print(f"STALE: {OUT_PATH} does not match a fresh generation run.\n"
                  f"Regenerate with: python tests/gen_golden.py", file=sys.stderr)
            return 1
        print(f"fresh: {OUT_PATH}")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)
    print(f"wrote {OUT_PATH} ({len(data['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
