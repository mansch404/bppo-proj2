# Advanced Custom Metric Appendix

This appendix documents the five custom metrics used to evaluate resource-allocation strategies.

## 1) Value-Weighted Wait (min)
- Formula: sum(wait_i * w_i) / sum(w_i) / 60, where w_i = log(1 + requested_amount_i).
- Why valuable: penalizes long waits on high-value cases more strongly than low-value cases.
- Example: two tasks each wait 30 minutes; amounts are 1,000 and 100,000. Plain average wait is equal, but this metric exposes the larger business impact of delaying the high-value case.

## 2) Value-at-Risk SLA Breach (%)
- Formula: 100 * (sum(requested_amount_i for wait_i > SLA) / sum(requested_amount_i)).
- Why valuable: quantifies how much business value is exposed to SLA violations, not just how many tasks are late.
- Example: Strategy A breaches SLA on 10 small cases, Strategy B breaches on 2 very large cases; count-based SLA looks better for B, but value-at-risk reveals higher downside.

## 3) Case Handover Rate
- Formula: average across cases of (resource switches / (tasks_in_case - 1)).
- Why valuable: captures continuity loss, coordination overhead, and context-switch friction per case.
- Example: two strategies have similar cycle times; one keeps ownership stable while the other frequently reassigns between resources. This metric distinguishes them.

## 4) Automation Leverage on Eligible Tasks (%)
- Formula: 100 * (eligible tasks handled by system resources / all eligible tasks).
- Why valuable: measures whether a strategy actually uses automation opportunities where available.
- Example: if both strategies have similar wait time but one offloads repetitive eligible tasks to bots, this metric shows better automation utilization.

## 5) Human Capacity Stress Ratio (%)
- Formula: 100 * (sum(overload_seconds) / sum(capacity_seconds)), with overload_seconds = max(0, used - capacity) per resource-day.
- Why valuable: detects hidden overcommitment that may not appear in short-term cycle-time metrics.
- Example: a strategy can produce low average wait by overloading a few humans well beyond daily capacity; this metric flags that operational risk.
