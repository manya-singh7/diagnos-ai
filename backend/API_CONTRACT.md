# Diagnos AI — REST API Contract Documentation

This document defines the interface contract for the **Smart Guided Troubleshooting Engine** service.
Frontend and client components can build against these specifications.

---

## 1. GET `/health`
Health readiness probe verifying service, LLM connectivity, and catalog loading.

### Response `200 OK`
```json
{
  "status": "ok",
  "model_ready": true,
  "catalog_ready": true
}
```

---

## 2. POST `/v1/troubleshoot`
Processes a customer complaint and optional customer-care reference text (`siis_response`), returning up to 2 ranked troubleshooting action plans.

### Request Body (`application/json`)
```json
{
  "query": "The mobile phone swipe navigation moves up or down instead of left or right after downloading an app",
  "siis_response": "Optional pre-cleaned customer-care reference text (untrusted passive data)"
}
```
* Note: Any web URLs in `query` or `siis_response` are programmatically stripped upon intake.

### Response `200 OK` (`ContextDeeplinkResponse`)
```json
{
  "contexts": [
    {
      "goal": "Follow these steps to perform this Swipe Navigation Troubleshooting",
      "title": "Swipe navigation settings",
      "score": 0.93,
      "actions": [
        {
          "actionName": "Configure Navigation Bar Settings",
          "description": "It will let you choose navigation type",
          "category": "auto",
          "stepGroups": [
            {
              "steps": [
                "Navigate to and open Settings.",
                "Tap on Display.",
                "Tap on Navigation bar.",
                "Select your preferred navigation type between Buttons and Swipe gestures.",
                "Optionally toggle on Gesture hint to display guidance lines at the bottom of the screen."
              ],
              "actionableDeeplink": {
                "deeplink": "bixby://masked/act/setting/display/navigation_bar",
                "description": "Open navigation bar settings under Display",
                "message": "choose navigation type in Display settings"
              },
              "validationDeeplink": null
            }
          ]
        },
        {
          "actionName": "Restart Device in Safe Mode",
          "description": "It will disable all third party apps",
          "category": "critical",
          "stepGroups": [
            {
              "steps": [
                "Press and hold the Power button.",
                "Touch and hold the Power off icon.",
                "Tap Safe mode to reboot and test gesture behavior."
              ],
              "actionableDeeplink": {
                "deeplink": "bixby://dummy_positive",
                "description": "Open general device settings placeholder",
                "message": "navigate to unindexed settings screen"
              },
              "validationDeeplink": null
            }
          ]
        }
      ]
    }
  ],
  "fallback": null,
  "meta": {
    "latency_ms": 185,
    "cache_hit": false,
    "model": "gemini-3.6-flash",
    "cost_usd": 0.000142
  }
}
```

### Fallback Response (When no match is viable)
```json
{
  "contexts": [],
  "fallback": "no_match",
  "meta": {
    "latency_ms": 95,
    "cache_hit": false,
    "model": "gemini-3.6-flash",
    "cost_usd": 0.0
  }
}
```

### Contract Constraints & Guarantees
1. **Ranked Hypotheses**: `contexts` contains up to 2 `Goal` objects ordered by confidence `score` descending.
2. **Category Ordering**: Actions inside every goal are ordered strictly: `auto` (non-invasive settings) first $\rightarrow$ `manual` (physical cleaning/hardware) $\rightarrow$ `critical` (reboot/factory reset/safe mode) last.
3. **Deeplink Integrity**:
   - `auto`: Carries catalog deeplink or `bixby://dummy_positive`.
   - `manual`: Strictly `actionableDeeplink = null`.
   - `critical`: Carries catalog deeplink or `bixby://dummy_positive`.
4. **Zero Web URLs**: Steps, descriptions, titles, and goals strictly contain **zero** `http`, `https`, `www.`, or markdown links.

---

## 3. POST `/v1/clarify`
Stateless endpoint for clarifying ambiguous queries or re-ranking results with a user's answer.

### Scenario A: Check If Clarification Is Needed (No Answer Provided)
Send original query and the top-2 hypotheses from the initial diagnosis.

#### Request Body
```json
{
  "query": "The phone screen gestures are not responding properly",
  "hypotheses": [
    { "title": "Swipe navigation settings", "score": 0.85 },
    { "title": "Touch screen sensitivity", "score": 0.80 }
  ],
  "clarification_answer": null,
  "gap_threshold": 0.15
}
```

#### Response `200 OK` (Clarification Question Needed, Score Gap $< 0.15$)
```json
{
  "contexts": [],
  "fallback": null,
  "meta": {
    "latency_ms": 1,
    "cache_hit": false,
    "model": "gemini-3.6-flash",
    "cost_usd": 0.0
  },
  "needs_clarification": true,
  "question": "Did this issue start with swipe navigation settings, or does it involve touch screen sensitivity?"
}
```

#### Response `200 OK` (No Clarification Needed, Score Gap $\ge 0.15$)
If score gap is $\ge 0.15$ (e.g. scores 0.90 and 0.65):
```json
{
  "contexts": [],
  "fallback": null,
  "meta": {
    "latency_ms": 1,
    "cache_hit": false,
    "model": "gemini-3.6-flash",
    "cost_usd": 0.0
  },
  "needs_clarification": false,
  "question": null
}
```

---

### Scenario B: Clarification Answer Provided
When the user submits an answer to the clarifying question:

#### Request Body
```json
{
  "query": "The phone screen gestures are not responding properly",
  "hypotheses": [
    { "title": "Swipe navigation settings", "score": 0.85 },
    { "title": "Touch screen sensitivity", "score": 0.80 }
  ],
  "clarification_answer": "It started after I downloaded a third party launcher app"
}
```

#### Response `200 OK`
Re-runs the pipeline with the enriched query, returning updated ranked `contexts` matching the standard troubleshooting schema with `needs_clarification: false`.
* Untrusted input protection: Embedded URLs in `clarification_answer` are scrubbed. Empty or garbage answers return a clean fallback rather than a 500 error.
