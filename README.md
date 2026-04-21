# Crypto-Trader Ver 6 Beta

Adaptive crypto trading bot with:
- regime detection
- strategy routing
- risk-based sizing
- break-even and trailing exits
- Telegram notifications
- PostgreSQL trade journal

This version is long-only and designed for spot-friendly execution.

## Control semantics (deterministic)

| State | Behavior |
|------|--------|
| enabled = true | normal trading |
| enabled = false + flatten=false | stop new entries, manage existing |
| enabled = false + flatten=true | immediately close all positions |

These semantics are enforced in the runtime patch layer (sitecustomize).

## Caffeine API contract

### GET `/caffeine/state`

Returns:
```
{
  "assets": {...},
  "controls": {...},
  "last_update": "ISO",
  "schema_version": 1
}
```

### POST `/caffeine/controls`

Request:
```
{
  "scope": "BTC/USDT",
  "enabled": false,
  "flatten_on_disable": true
}
```

Response:
```
{
  "ok": true,
  "controls": {...full snapshot...},
  "schema_version": 1
}
```

## Testing

Basic invariant tests are included using pytest:

```
pip install -r requirements.txt
pytest
```

Covers:
- position sizing caps
- TP/SL lifecycle safety
- startup duplication guard

## Caffeine dashboard integration

Set these environment variables in Render:

- `CAFFEINE_URL`
- `CAFFEINE_TOKEN` (optional)
- `ALLOWED_ORIGINS` (optional)

The bot logs non-2xx responses from Caffeine for debugging.
