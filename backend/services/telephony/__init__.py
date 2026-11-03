"""Phone providers: one agent, several carriers.

`base` defines what a provider is (its accounts, how it places a call, how it reads a refusal) and what its audio
WebSocket looks like (`StreamCodec`); `exotel`, `twilio`, `plivo` (also Vobiz) and `telnyx` implement them; `registry`
chooses the order in which providers and accounts are tried for one call (PHONE_PROVIDERS, PHONE_ROUTES) and `state`
remembers which account is out of credit. The call itself (voice activity, turns, barge-in) is routers/phone_exotel.py,
which is the same for every provider: it speaks to the carrier only through a codec."""
