"""Shared feature construction — used by both training and serving so the text the
model sees is identical in both paths."""

META_FIELDS = [
    "channel", "merchant_vertical", "currency", "dispute_value_band",
    "chargeback_cycle", "merchant_country", "cardholder_region",
]


def _days_bucket(days):
    try:
        d = float(days)
    except (TypeError, ValueError):
        return "days_unknown"
    if d <= 7:
        return "days_0_7"
    if d <= 30:
        return "days_8_30"
    if d <= 60:
        return "days_31_60"
    if d <= 90:
        return "days_61_90"
    if d <= 120:
        return "days_91_120"
    return "days_over_120"


def build_text(row: dict) -> str:
    """Serialize a dispute record into a single string: narrative first, then
    metadata as explicit `field=value` tokens (helps the linear model treat them
    as discrete signals rather than free prose)."""
    parts = [str(row.get("dispute_narrative", "") or "")]
    for f in META_FIELDS:
        v = row.get(f)
        if v is not None and str(v).strip():
            parts.append(f"{f}={str(v).strip().lower().replace(' ', '_')}")
    parts.append(_days_bucket(row.get("days_since_transaction")))
    amt = row.get("transaction_amount")
    try:
        amt = float(amt)
        if amt < 25:
            band = "amt_lt_25"
        elif amt < 100:
            band = "amt_25_100"
        elif amt < 500:
            band = "amt_100_500"
        elif amt < 2000:
            band = "amt_500_2000"
        else:
            band = "amt_gt_2000"
        parts.append(band)
    except (TypeError, ValueError):
        pass
    return "  ".join(parts)
