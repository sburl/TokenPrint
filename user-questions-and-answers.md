**Created:** 2026-03-03-17-45
**Last Updated:** 2026-03-03-17-45

# TokenPrint - User Questions and Answers

## Why this repo exists (current best understanding)
TokenPrint is a developer-first analytics tool for local AI CLI usage. It currently:
- gathers daily usage from Claude Code (`ccusage`), Codex (`ccusage-codex`), and Gemini (`~/.gemini/telemetry.log`),
- computes daily totals and estimated costs, energy, carbon, and water usage,
- renders a single-page interactive dashboard in a bundled HTML report,
- can run once (`tokenprint`) or continuously via a Go daemon (`tokenprintd`).

Yeah, that's right. It's to see how many tokens you're using and then just to give you an idea of what the impact of that is, and then to be pretty easily shareable so you can have people check in.

## Existing shape and constraints I inferred
- Python package lives in `tokenprint/` with most logic centralized in `tokenprint/__init__.py`.
- A fairly comprehensive test file already exists: `test_tokenprint.py`.
- Go side is a lightweight HTTP server in `daemon/go/main.go` with endpoint tests in `daemon/go/main_test.go`.
- No CI config exists in this repo yet (`.github` is absent).
- No open PRs surfaced from GitHub CLI in this run.
- No local PR review files (`.github/pull_request_template.md`, etc.) currently exist.
- There is an existing CrossCheck setup on this machine at `/Users/sba/.claude/CrossCheck`, but no dedicated Gemini-PR script is present in this repository.

Yeah, so this is auto-installed when people install crosscheck, just to help them monitor all their tokens. I think there's a lot of good stuff here. This is currently a public repo, so you do need to watch what you put out. This doc should not make it into the repo and stuff like that, but I wouldn't worry too much about it. I just want to make the best experience possible.

## What I propose to treat as high-confidence truths
- Primary goal is quality restoration and maintainability first, not feature velocity.
- You want frequent, small PRs with explicit PR docs and strict review loops.
- You want external verification beyond local tests (security/bug scans and AI review).
- You want the daemon + one-shot CLI both to stay stable while we refactor.
- You want no further user questions after this baseline question round.

We want to have accuracy as the number one goal. We want everything to be really accurate. I think also we want to focus on having a fairly beautiful UI before we really flesh out all the features. Maintainability is important on the codebase too, but all these other things are right.

## Questions for you (please answer in this order)
1. Primary audience for TokenPrint: only me, small team, or public users?

This is a public repo.I don't think that many people use it, but we should act as though they do.

2. Which release behavior matters most right now: daily reporting accuracy, developer UX, or architecture modernization?
Accuracy is always the number one thing. I think the UX for someone using this is nice too.
3. Do you want backwards compatibility for generated dashboard HTML output format and structure?
No, I think no one's really using this. You don't need to worry about backwards compatibility ever.
4. Should `tokenprint` remain a single-command CLI, or can we split into subcommands (e.g., `tokenprint collect`, `tokenprint render`)?
I would prefer for it to be a single command line thing, because then that would be easier. We could also think about how to distribute this on NPM or Bun or any of those other things.
5. Is exact numerical parity with current calculations mandatory, or can environmental models/constants change if documented?
No, we need to just calculate what token usage is, and you should be focused on accurately computing that.Where's our computer today? Maybe we make some errors today, and we should always be focused on accuracy above everything else.
6. Should cached tokens treatment be revised for Gemini/Codex if model data quality changes in upstream logs?
I'm not sure how to deal with these cache token things. I think it would be ideal to break them out because I think it's pretty messy right now.Yeah, in general I think we shouldn't think about cash tokens as much because no one's paying for them or anything.
7. Do you want provider defaults to stay exactly Claude/Codex/Gemini, or do we add a plugin model for future providers now?
Open router support eventually.
8. Are local filesystem writes (cache + temp HTML) always okay, or should all data be optional and fully in-memory mode be added?
I don't understand what this means. I'll defer to you. It's nice for things to stay in the app rather than being all over someone's machine.
9. Is remote/CI execution allowed to fetch `npm`/`go`/`python` dependency outputs in non-interactive mode?
I don't think we do that right now. I don't understand this question.
10. For Gemini logs, is reading by file path user-defined or should it remain fixed to `~/.gemini/telemetry.log`?
I don't understand this question. You should just find a way to get Gemini logs in the most accurate, maintainable way possible.
11. Are we okay deleting stale cache files/format migrations automatically, or must cache be strictly preserved between upgrades?
I think it's not that big a deal. I think we do want to kind of preserve cache so a user can see their history. Maybe there needs to be some kind of upgrade thing there, but it's not that big a deal.
12. How strict should daemon security be for `/api/refresh` in LAN vs public host scenarios?
I have no idea; make a decision.
13. Should refresh endpoint token be required even on loopback in your policy?
No idea.
14. Is browser auto-open still expected, or should we make it opt-in only in CLI defaults?
I don't understand this question.
15. Can CI run non-network-reliant unit tests and linting on every PR, with heavier integration/e2e nightly?
Yeah, that's fine.
16. Should PRs be generated against short-lived branches with squash merges only, or do you want merge commit/squash choice documented by branch?
Branches are ideal.
17. Is any provider-specific rate/cost model currently approved externally or should we make constants configurable via file/env?
I think we got all the constants externally. If there's a new model that comes out and it's not supported, we might just guess what the price will be, but I think you should just go ahead and try to get as much of the data externally as possible. I think we have a provider installed for that even.
18. Do you want chart/interactions/theme refactors now, or only logic and infrastructure in this phase?
I think the charts that we have right now look pretty good and they're pretty nice. You could think about other ways to show the data or something, but I'm kind of struggling on what more to pull out here.
19. Do we need to preserve the current HTML font + color palette, or is full UI restyle acceptable?
You could restore the full UI if you want. I kind of like it, but if you think you can do something better, you can take a shot at it.
20. How much breakage tolerance for test additions: additive-only vs strict expected snapshots of existing behavior?
I don't understand this.
21. Should we enforce command usage via a `python -m tokenprint` entry as the canonical path?
If you don't understand this, I'll defer to you.
22. Any external API keys or secret-handling requirements when running CI from GitHub?
I don't think so. I'll defer to you, though.
23. Should security posture favor strict defaults (e.g., refuse to run without refresh token) by default, or backward compatibility now?
I don't think that this really has that. I'm kind of confused about what you're doing.
24. Do you want to keep `template.html` as-is and only inject data, or restructure to a static template system now?
I'm kind of confused about this. I'll defer to you.
25. Are you okay with moving to smaller modules/files before feature additions (explicit refactor-first) or feature-first in existing file?
Yeah, that's fine. I think it would be good to refactor and make it extendable first.
26. Any deadlines or milestones for feature categories (quick wins vs long-term vision)?
Nope, I thought this was pretty built. I'm curious what you're going to do with it.

## Current non-code observations
- The Python parser for provider data is permissive and already defensive, but edge cases still exist around timestamp/date formats and cache merge correctness.
- The go daemon is intentionally simple and currently exposes both same-origin and no-origin API calls for refresh.
- Security posture for local endpoints is decent for localhost but not hardened for broader exposure.
- Test coverage is strongest in Python but there are still behavior areas with implicit assumptions (cache schema evolution, malformed logs, cross-provider normalization).
- Without CI this repo is susceptible to regressions from local-only confidence.

## What I need before broad refactor
Please answer all 26 items above (or confirm “previous assumptions hold”) so the first full implementation stage can start without follow-up questions.
