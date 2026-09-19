# Northwind Fleet Management API Reference

Version 3.4 — published August 2025

The Fleet Management API lets you programmatically control and monitor a fleet of
Northwind Robotics robots, including the Atlas AR-200 and Orion OR-500. This
reference covers authentication, the task, robot, and charging endpoints, error
handling, rate limits, and webhooks.

The API base URL is `https://api.northwind-example.com/v3`. All requests must be
made over HTTPS. Requests made over plain HTTP are rejected with status 400.

---

## Authentication

The Fleet API uses bearer-token authentication. Every request must include an
`Authorization` header containing an API key issued from the fleet management
console:

```
Authorization: Bearer nw_live_XXXXXXXXXXXXXXXX
```

API keys are created in the console under Account → API Keys. A key created there
begins with the prefix `nw_live_` for production and `nw_test_` for the sandbox
environment. A single account may hold up to 10 active keys at a time. Revoking a
key takes effect immediately.

Keys created before version 3.0 used the prefix `nw_key_` and are now deprecated.
Deprecated keys continue to work until 31 December 2025, after which they will be
rejected with status 401.

If an API key is missing or invalid, the API responds with status 401 and an error
body of the form `{"error": "invalid_api_key"}`. If a valid key lacks permission
for the requested action, the API responds with status 403 and
`{"error": "insufficient_scope"}`.

---

## Rate Limits

The Fleet API enforces a rate limit of 120 requests per minute per API key. The
current limit state is returned in every response through three headers:
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`. The reset
header gives the number of seconds until the window resets.

When the rate limit is exceeded, the API responds with status 429 and a
`Retry-After` header indicating how many seconds to wait before retrying. Clients
should honour this header rather than retrying immediately. Batch endpoints, marked
below, count as a single request regardless of how many items they contain.

---

## Robots

### List robots

`GET /robots` returns all robots in the fleet. Each robot object contains its
`id`, `model` (either `atlas-ar200` or `orion-or500`), `status`, `battery_percent`,
and `current_task_id`. The `status` field is one of `idle`, `working`,
`charging`, `error`, or `offline`.

You can filter by model with the query parameter `?model=orion-or500` and by status
with `?status=error`. The two filters may be combined.

### Get a single robot

`GET /robots/{robot_id}` returns one robot. If the robot ID does not exist, the API
responds with status 404 and `{"error": "robot_not_found"}`.

### Robot health

`GET /robots/{robot_id}/health` returns diagnostic information including the last
reported error code, firmware version, and total operating hours. An Atlas AR-200
running current firmware reports version `4.7.1`; an Orion OR-500 reports `2.9.3`.
The error code field is null when the robot has no active fault.

---

## Tasks

### Create a task

`POST /tasks` creates a new transport task and assigns it to an available robot.
The request body must include a `pickup` location and a `dropoff` location, each
given as a named waypoint from the warehouse map. You may optionally specify a
`model` to require a particular robot type; for example, a pallet task should set
`"model": "orion-or500"` because the Atlas AR-200 cannot carry pallets.

If no suitable robot is available, the task is queued and its `status` is returned
as `pending`. When a robot is assigned, the status becomes `assigned`, then
`in_progress`, and finally `completed`. A task that cannot be completed becomes
`failed`, and the failure reason is given in the `failure_reason` field.

A task created without a `model` field is eligible for any robot type. A task that
specifies a payload above 45 kilograms will never be assigned to an Atlas AR-200,
even if no model is specified, because the Atlas payload limit is 45 kilograms.

### Cancel a task

`DELETE /tasks/{task_id}` cancels a task. A task that is already `completed` cannot
be cancelled; attempting to do so returns status 409 and
`{"error": "task_already_completed"}`. A task that is `in_progress` is cancelled
gracefully: the robot completes its current motion and then stops safely.

### Batch create

`POST /tasks/batch` creates up to 50 tasks in a single request. This endpoint
counts as one request against the rate limit. If any task in the batch is invalid,
the entire batch is rejected with status 422 and a per-item list of errors; no
tasks in the batch are created.

---

## Charging

`POST /robots/{robot_id}/charge` sends a robot to its charging dock immediately,
interrupting any current task. This is useful before a planned network maintenance
window. A robot already charging returns status 200 with no change.

Note that the Atlas AR-200 and Orion OR-500 use incompatible charging docks, so a
robot can only be sent to a dock of its own model. Sending a robot to charge when
no compatible dock is free returns status 409 and `{"error": "no_dock_available"}`.

---

## Webhooks

The Fleet API can notify your systems of events through webhooks. Register a
webhook endpoint in the console under Account → Webhooks. Northwind will send a
`POST` request to your endpoint whenever a subscribed event occurs.

The available event types are `task.completed`, `task.failed`, `robot.error`,
and `robot.offline`. Each webhook payload includes an `event` field, a `timestamp`
in ISO 8601 format, and a `data` object describing the event.

Every webhook request is signed with an HMAC-SHA256 signature in the
`X-Northwind-Signature` header. You should verify this signature using your
webhook signing secret, which is shown once when the webhook is created. If you do
not verify the signature, a malicious actor could send forged events to your
endpoint.

Northwind retries a failed webhook delivery up to 5 times with exponential backoff.
If all retries fail, the event is dropped and recorded in the console webhook log.
A webhook endpoint must respond with a 2xx status within 10 seconds, or the
delivery is considered failed.

---

## Versioning and Deprecation

The API version is part of the URL path, currently `v3`. Northwind supports each
major version for at least 18 months after the next major version is released.
Version 2 of the Fleet API reached end of life on 30 June 2025 and no longer
accepts requests. Clients still calling `v2` receive status 410 with
`{"error": "version_retired"}`.

Non-breaking changes, such as new fields on existing responses, are added to the
current version without a version bump. Clients should therefore ignore unknown
fields rather than treating them as errors.

For questions about the API that are not covered here, contact the developer
support team at api-support@northwind-example.com.
