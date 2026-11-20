---
name: review-model-change
description: Review a pull request that changes which model the support agent uses (routing, defaults, fallbacks) or that claims to fix a Braintrust pattern. Use before merging any change to model selection, src/tau2/config.py model settings, or src/tau2/agent/.
---

# Review a model or routing change

Review only. Don't edit files or push.

## 1. Check the change against production traces

If the PR or the user links a Braintrust pattern, open at least one evidence
trace with the `bt view` command it gives. For each trace you open:

- Say what the customer asked for and which tools the agent called.
- Follow the PR's code path for that request and say which model it would use.
  If the trace would not reach the new code path, that is a blocking issue.

## 2. Team rules for model changes

Flag each rule the PR breaks.

1. **Eval first.** The PR links an eval that compares quality and cost of the
   old and new model on the affected traffic. Any quality drop is stated in the
   PR, with who accepted it.
2. **Measurable in Braintrust.** Every request records which model answered it
   and why (for example `metadata.routed_model`, `metadata.route_reason`) and
   its cost (`metadata.cost_usd`), on the trace, not only in local logs.
   Without this we can't confirm the pattern went away after release.
3. **Count every call.** Extra LLM calls the change adds (classifiers, retries)
   are included in the cost estimate.
4. **Safe fallback.** Failures fall back to the configured model, and the
   fallback is tested. Completed order changes are never replayed.
5. **Check the tools it will call.** If the new model will call tools that change
   orders (`modify_pending_order_items`, `exchange_delivered_order_items`,
   `cancel_pending_order`, `return_delivered_order_items`), read those tools in
   `src/tau2/domains/retail/tools.py` and flag known bugs on that path.
6. **Off switch.** The change can be turned off in `src/tau2/config.py` without
   a code change.

## 3. Report

List issues by severity (High, Medium, Low), each with file and line, the rule
or trace it comes from, and a suggested fix. End with **Blocks merge: yes/no**
and the issues that block it.
