# BonClay verify SDK

Check an BonClay receipt, a clip's hash and the public withdrawal log **without an BonClay server**. The format is in
[`docs/RECEIPT_SPEC.md`](../../docs/RECEIPT_SPEC.md). Python needs `cryptography`; the JavaScript file needs Node 18+
and nothing else.

```python
from bonclay_verify import verify_receipt, verify_audio_sha256, verify_withdrawal_log
ok = verify_receipt(receipt, public_key_b64)
```
```js
import { verifyReceipt } from "./verify.mjs";
```

Get `public_key_b64` from `GET /.well-known/bonclay-receipt-key` of the server that made the clip, over a channel you
trust. A valid signature proves the key holder issued the receipt; it does not prove who the key holder is.
