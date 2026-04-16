# Anonymous Publishing Guide

This document describes how to prepare and upload the anonymous version of LLM-Para to the GitHub secondary account (llmpara2026) for anonymous review.

**Anonymous repo:** https://github.com/llmpara2026/LLM-Para

## Files to EXCLUDE from the anonymous repository

The following files/directories should NOT be pushed to the anonymous repo:

```
paper_draft/                  # Contains paper source, author info, figures
DEPLOY.md                     # Contains original GitHub username and deployment details
RESEARCH_DOC.md               # Contains original GitHub links and author contact info
ANONYMOUS_PUBLISH_GUIDE.md    # This guide itself
google89265ef569fd1ea7.html   # Google Search Console verification (can identify owner)
.git/                         # Git history may contain author info in commits
.claude/                      # Claude Code session data
```

## Google Search Console verification in index.html

The file `static/index.html` contains a Google verification meta tag (line 6) that could be traced to the site owner:
```html
<meta name="google-site-verification" content="hpjjlSqoIlrXEiQ1JxYm0EA_YjL_vuPzqczKiLixfVY" />
```
**Recommend removing this line from the anonymous version.**

## Files that have been modified for anonymous review

The following files have been edited to remove identifying information:

1. **`static/index.html`** — GitHub button commented out (search for `[ANONYMOUS REVIEW]`)
2. **`README.md`** — Author name → "Anonymous Authors", GitHub URLs → llmpara2026 account, paper_draft references hidden
3. **`paper_draft/llm_para_paper.tex`** — GitHub URLs replaced (but this folder is excluded anyway)
4. **`paper_draft/PAPER_DRAFT.md`** — GitHub URLs replaced (but this folder is excluded anyway)

## How to push to anonymous GitHub account

```bash
# 1. Create a clean copy (exclude sensitive files)
mkdir /tmp/LLM-Para-anon
rsync -av --exclude='paper_draft/' \
          --exclude='DEPLOY.md' \
          --exclude='RESEARCH_DOC.md' \
          --exclude='ANONYMOUS_PUBLISH_GUIDE.md' \
          --exclude='google89265ef569fd1ea7.html' \
          --exclude='.git/' \
          --exclude='.claude/' \
          . /tmp/LLM-Para-anon/

# 2. Remove Google verification meta tag from the copy
sed -i '/<meta name="google-site-verification"/d' /tmp/LLM-Para-anon/static/index.html

# 3. Initialize and push to the anonymous account
cd /tmp/LLM-Para-anon
git init
git add .
git commit -m "Initial release: LLM-Para analytical framework"
git remote add origin https://github.com/llmpara2026/LLM-Para.git
git branch -M main
git push -u origin main
```

## After paper acceptance — restoring original content

Search for `[ANONYMOUS REVIEW]` across the codebase to find all commented-out sections:

```bash
grep -r "ANONYMOUS REVIEW" --include="*.html" --include="*.md" --include="*.tex"
```

Restore:
1. `static/index.html` — Uncomment the GitHub button
2. `README.md` — Restore author name, original GitHub URLs, paper badges, key figures section
3. `paper_draft/llm_para_paper.tex` — Restore original GitHub URLs
4. `paper_draft/PAPER_DRAFT.md` — Restore original GitHub URLs

## Important notes

- The live demo at https://llm-para.onrender.com does NOT reveal author identity (GitHub button is hidden)
- The Render deployment URL is generic and does not contain author information
- Third-party project links in README (LLM-Viewer, LLMCompass, FlashAttention) are kept as-is since they reference other authors' work
- Make sure the llmpara2026 GitHub profile does not have your real name or institution listed
