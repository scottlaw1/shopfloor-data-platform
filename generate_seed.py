"""Generate deliberately messy raw seed data for the shopfloor data platform.

Fakes five raw sources for a barbershop SaaS and lands them in data/raw/:

    shops.csv             admin/CRM export
    barbers.csv           staff roster export
    customers.csv         booking-app signup export
    appointments.csv      booking-system export (no-shows, reschedules, tips)
    billing_events.jsonl  payment-processor webhook log (subscriptions, refunds, failed charges)
    crm_contacts.csv      marketing CRM export whose emails/phones don't line up with booking

Clean records are generated first, then explicit mess_* / inject_* functions
corrupt them on purpose (duplicates, nulls, timezone mismatches, format noise)
so downstream dbt models have real cleanup work to do. See README.md for the
full list of injected defects.

The true CRM-contact -> booking-customer mapping goes to
data/answer_key/crm_customer_matches.csv so entity-resolution logic can be scored.

Usage:
    uv run python generate_seed.py [--seed 42] [--out data/raw] [--shops 12]
                                   [--customers 2000] [--days 270]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from faker import Faker

UTC = timezone.utc

# Fixed anchor so output depends only on --seed, never on the day you run it.
# The default 270-day window spans both the March and November DST transitions.
END_DATE = date(2026, 11, 7)

# The booking system's legacy app server writes updated_at in its own local time.
LEGACY_SERVER_TZ = ZoneInfo("America/Chicago")

# (city, state, zip prefix, IANA timezone). Real zip prefixes so leading zeros show up.
LOCATIONS = [
    ("Boston", "MA", "021", "America/New_York"),
    ("Newark", "NJ", "071", "America/New_York"),
    ("Brooklyn", "NY", "112", "America/New_York"),
    ("Atlanta", "GA", "303", "America/New_York"),
    ("Miami", "FL", "331", "America/New_York"),
    ("Chicago", "IL", "606", "America/Chicago"),
    ("Houston", "TX", "770", "America/Chicago"),
    ("Austin", "TX", "787", "America/Chicago"),
    ("Denver", "CO", "802", "America/Denver"),
    ("Phoenix", "AZ", "850", "America/Phoenix"),
    ("Tucson", "AZ", "857", "America/Phoenix"),
    ("Los Angeles", "CA", "900", "America/Los_Angeles"),
    ("Portland", "OR", "972", "America/Los_Angeles"),
    ("Seattle", "WA", "981", "America/Los_Angeles"),
]

PLAN_TIERS = ["basic", "pro", "enterprise"]
PLAN_TIER_WEIGHTS = [0.5, 0.35, 0.15]
PLAN_PRICES = {"basic": 4900, "pro": 9900, "enterprise": 19900}

# service -> (base price cents, duration minutes)
SERVICES = {
    "haircut": (3500, 30),
    "skin_fade": (4000, 45),
    "beard_trim": (2000, 20),
    "hot_towel_shave": (3500, 30),
    "facial": (4500, 40),
    "cut_and_beard": (5500, 60),
}
SERVICE_NAMES = list(SERVICES)
SERVICE_WEIGHTS = [0.33, 0.20, 0.12, 0.08, 0.10, 0.17]

# weekday -> (open hour, close hour, mean appointments per barber) in shop-local time.
SHOP_HOURS = {
    0: None,  # closed Mondays
    1: (9, 19, 1.3),
    2: (9, 19, 1.3),
    3: (9, 19, 1.3),
    4: (9, 19, 1.6),
    5: (8, 17, 2.2),
    6: (10, 16, 1.0),
}
GAP_MINUTES = [0, 15, 30, 45, 60, 90]
GAP_WEIGHTS = [0.25, 0.25, 0.2, 0.12, 0.1, 0.08]

STATUSES = ["completed", "no_show", "cancelled", "rescheduled"]
STATUS_WEIGHTS = [0.80, 0.08, 0.05, 0.07]
MAX_RESCHEDULE_DEPTH = 2
NO_SHOW_FEE_CENTS = 1000

PAYMENT_METHODS = ["card", "cash", "app"]
PAYMENT_WEIGHTS = [0.60, 0.28, 0.12]

FAILURE_CODES = ["card_declined", "insufficient_funds", "expired_card", "processing_error"]
REFUND_REASONS = ["requested_by_customer", "duplicate", "service_issue"]
SUBSCRIPTION_FAILURE_RATE = 0.06
CHARGE_FAILURE_RATE = 0.06
REFUND_RATE = 0.03

# Spellings a "null" takes on depending on which upstream form/export touched the row.
NULL_TOKENS = [None, "NULL", "N/A", " "]

STATUS_VARIANTS = {
    "no_show": ["No-Show", "NOSHOW", "no show"],
    "completed": ["Completed", "COMPLETED"],
    "cancelled": ["canceled", "Cancelled"],
}

# The CRM portal is configured for the company's HQ timezone and exports naive local time.
CRM_PORTAL_TZ = ZoneInfo("America/New_York")
LEAD_SOURCES = ["instagram", "google", "walk_in", "referral", "yelp"]
NICKNAMES = {
    "William": "Bill", "Robert": "Bob", "Michael": "Mike", "James": "Jim", "Richard": "Rick",
    "Joseph": "Joe", "Thomas": "Tom", "Christopher": "Chris", "Daniel": "Dan", "Matthew": "Matt",
    "Anthony": "Tony", "Benjamin": "Ben", "Nicholas": "Nick", "Samuel": "Sam", "Andrew": "Andy",
    "Edward": "Ed", "Jennifer": "Jen", "Elizabeth": "Liz", "Katherine": "Kate", "Jessica": "Jess",
}
NICKNAMES |= {nick: full for full, nick in NICKNAMES.items()}
EMAIL_DOMAIN_TYPOS = {
    "gmail.com": ["gmial.com", "gmail.co", "gmai.com"],
    "yahoo.com": ["yaho.com", "yahoo.con"],
    "hotmail.com": ["hotmail.con", "hotmial.com"],
}


# --------------------------------------------------------------------------- helpers


def make_id(rng: np.random.Generator, prefix: str, nbytes: int = 8) -> str:
    return f"{prefix}_{rng.bytes(nbytes).hex()}"


def email_for(first: str, last: str, domain: str, suffix: str = "") -> str:
    local = re.sub(r"[^a-z.]", "", f"{first}.{last}".lower())
    return f"{local}{suffix}@{domain}"


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def vary_email(rng: np.random.Generator, email: str) -> str:
    local, domain = email.split("@")
    variants = [
        f"{local.capitalize()}@{domain.upper()} ",
        email.upper(),
        f"{local}+cuts@{domain}",
        f" {email}",
    ]
    return variants[rng.integers(len(variants))]


def phone_digits(phone: str) -> str:
    """Last 10 digits of a phone number, ignoring any extension."""
    return re.sub(r"\D", "", phone.split("x")[0])[-10:]


def reformat_phone(rng: np.random.Generator, phone: str) -> str:
    digits = phone_digits(phone)
    a, b, c = digits[:3], digits[3:6], digits[6:]
    formats = [f"{a}{b}{c}", f"+1{a}{b}{c}", f"{a}-{b}-{c}", f"({a}) {b}-{c}", f"1.{a}.{b}.{c}"]
    return formats[rng.integers(len(formats))]


def shop_name(fake: Faker, rng: np.random.Generator, city: str) -> str:
    styles = [
        lambda: f"{fake.last_name()}'s Barbershop",
        lambda: f"The {rng.choice(['Sharp', 'Golden', 'Classic', 'Corner', 'Velvet'])} Chair",
        lambda: f"{city} Fade Co.",
        lambda: f"{fake.first_name()} & {rng.choice(['Blade', 'Razor', 'Comb'])}",
    ]
    return styles[rng.integers(len(styles))]()


def null_tokens(rng: np.random.Generator, n: int) -> list:
    return [NULL_TOKENS[i] for i in rng.integers(len(NULL_TOKENS), size=n)]


# --------------------------------------------------------------------------- clean builders


def build_shops(fake: Faker, rng: np.random.Generator, n_shops: int, start: date, end: date) -> list[dict]:
    picks = [LOCATIONS[i] for i in rng.permutation(len(LOCATIONS))]
    picks = (picks * (n_shops // len(picks) + 1))[:n_shops]
    # Always include a no-DST shop so fixed-offset assumptions break somewhere.
    if not any(loc[3] == "America/Phoenix" for loc in picks):
        picks[-1] = next(loc for loc in LOCATIONS if loc[3] == "America/Phoenix")

    shops = []
    for i, (city, state, zip_prefix, tz) in enumerate(picks, start=1):
        if rng.random() < 0.8:
            opened = start - timedelta(days=int(rng.integers(200, 3000)))
        else:
            opened = start + timedelta(days=int(rng.integers(10, max(11, (end - start).days // 2))))
        shops.append(
            {
                "shop_id": f"shp_{i:03d}",
                "name": shop_name(fake, rng, city),
                "address": fake.street_address(),
                "city": city,
                "state": state,
                "zipcode": f"{zip_prefix}{int(rng.integers(0, 100)):02d}",
                "timezone": tz,
                "opened_at": opened,
                "plan_tier": str(rng.choice(PLAN_TIERS, p=PLAN_TIER_WEIGHTS)),
                "phone": fake.phone_number(),
            }
        )
    return shops


def build_barbers(fake: Faker, rng: np.random.Generator, shops: list[dict], end: date) -> list[dict]:
    barbers = []
    for shop in shops:
        opened = shop["opened_at"]
        span = max((end - opened).days - 30, 1)
        for i in range(int(rng.integers(4, 9))):
            terminated = None
            if i < 2:  # founding barbers stay for the whole history
                hired = opened
            else:
                hired = opened + timedelta(days=int(rng.integers(0, span)))
                if rng.random() < 0.15 and (end - hired).days > 90:
                    terminated = hired + timedelta(days=int(rng.integers(60, (end - hired).days)))
            first, last = fake.first_name(), fake.last_name()
            on_commission = rng.random() < 0.7
            barbers.append(
                {
                    "barber_id": f"brb_{len(barbers) + 1:04d}",
                    "shop_id": shop["shop_id"],
                    "first_name": first,
                    "last_name": last,
                    "email": email_for(first, last, fake.free_email_domain()),
                    "phone": fake.phone_number(),
                    "hired_at": hired,
                    "terminated_at": terminated,
                    # Barbers are either on commission or rent their chair, never both.
                    "commission_pct": float(rng.choice([0.40, 0.45, 0.50, 0.55, 0.60])) if on_commission else None,
                    "chair_rent_cents": int(rng.integers(150, 301)) * 100 if not on_commission else None,
                }
            )
    return barbers


def build_customers(fake: Faker, rng: np.random.Generator, n: int, shops: list[dict]) -> list[dict]:
    customers = []
    for i in range(1, n + 1):
        first, last = fake.first_name(), fake.last_name()
        suffix = str(int(rng.integers(1, 100))) if rng.random() < 0.4 else ""
        customers.append(
            {
                "customer_id": f"cus_{i:06d}",
                "first_name": first,
                "last_name": last,
                "email": email_for(first, last, fake.free_email_domain(), suffix),
                "phone": fake.phone_number(),
                "home_shop_id": shops[rng.integers(len(shops))]["shop_id"],
                "created_at": None,  # backfilled from first booking in assign_customer_signups
                "marketing_opt_in": bool(rng.random() < 0.4),
            }
        )
    return customers


def add_appointment(
    rng: np.random.Generator,
    appts: list[dict],
    shop: dict,
    barber_id: str,
    customer_id: str,
    service: str,
    start_utc: datetime,
    booked_at: datetime,
    now: datetime,
    price_mult: float,
    rescheduled_from: str | None = None,
    depth: int = 0,
) -> None:
    base_price, duration = SERVICES[service]
    price = int(round(base_price * price_mult / 500) * 500)
    end_utc = start_utc + timedelta(minutes=duration)

    if end_utc > now:
        status = "booked"
    else:
        status = str(rng.choice(STATUSES, p=STATUS_WEIGHTS))
        if status == "rescheduled" and depth >= MAX_RESCHEDULE_DEPTH:
            status = "completed"

    appt = {
        "appointment_id": f"apt_{len(appts) + 1:07d}",
        "shop_id": shop["shop_id"],
        "barber_id": barber_id,
        "customer_id": customer_id,
        "service": service,
        "scheduled_start": start_utc,
        "duration_min": duration,
        "status": status,
        "rescheduled_from_id": rescheduled_from,
        "price_cents": price,
        "tip_cents": 0,
        "payment_method": None,
        "booked_at": booked_at,
        "updated_at": booked_at,
    }
    appts.append(appt)

    if status == "completed":
        method = str(rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS))
        tip = int(round(price * rng.uniform(0.10, 0.25) / 100) * 100) if rng.random() < 0.7 else 0
        if method == "cash" and rng.random() < 0.3:
            tip = None  # cash tips often never get keyed in
        appt.update(payment_method=method, tip_cents=tip, updated_at=end_utc + timedelta(minutes=int(rng.integers(0, 20))))
    elif status == "no_show":
        fee = NO_SHOW_FEE_CENTS if rng.random() < 0.5 else 0
        appt.update(price_cents=fee, payment_method="card" if fee else None, updated_at=start_utc + timedelta(minutes=30))
    elif status == "cancelled":
        appt.update(price_cents=0, updated_at=booked_at + (start_utc - booked_at) * rng.uniform(0.1, 0.99))
    elif status == "booked":
        appt.update(tip_cents=None)
    elif status == "rescheduled":
        rebooked_at = booked_at + (start_utc - booked_at) * rng.uniform(0.1, 0.95)
        appt.update(price_cents=0, updated_at=rebooked_at)

        tz = ZoneInfo(shop["timezone"])
        local = start_utc.astimezone(tz)
        new_day = local.date() + timedelta(days=int(rng.integers(1, 15)))
        if SHOP_HOURS[new_day.weekday()] is None:
            new_day += timedelta(days=1)
        # Keep the same local clock time, so the UTC time shifts if DST changed in between.
        new_start = datetime.combine(new_day, local.time(), tz).astimezone(UTC)
        add_appointment(
            rng, appts, shop, barber_id, customer_id, service, new_start, rebooked_at, now, price_mult,
            rescheduled_from=appt["appointment_id"], depth=depth + 1,
        )


def build_appointments(
    rng: np.random.Generator,
    shops: list[dict],
    barbers: list[dict],
    customers: list[dict],
    start: date,
    end: date,
    now: datetime,
) -> list[dict]:
    customers_by_shop = defaultdict(list)
    for c in customers:
        customers_by_shop[c["home_shop_id"]].append(c["customer_id"])
    all_customer_ids = [c["customer_id"] for c in customers]
    barbers_by_shop = defaultdict(list)
    for b in barbers:
        barbers_by_shop[b["shop_id"]].append(b)
    price_mult = {s["shop_id"]: rng.uniform(0.9, 1.3) for s in shops}

    appts: list[dict] = []
    day = start
    while day <= end:
        hours = SHOP_HOURS[day.weekday()]
        for shop in shops if hours else []:
            if day < shop["opened_at"]:
                continue
            open_h, close_h, rate = hours
            tz = ZoneInfo(shop["timezone"])
            close = datetime.combine(day, time(close_h), tz)
            regulars = customers_by_shop.get(shop["shop_id"]) or all_customer_ids
            for barber in barbers_by_shop[shop["shop_id"]]:
                if day < barber["hired_at"] or (barber["terminated_at"] and day > barber["terminated_at"]):
                    continue
                cursor = datetime.combine(day, time(open_h), tz)
                for _ in range(rng.poisson(rate)):
                    cursor += timedelta(minutes=int(rng.choice(GAP_MINUTES, p=GAP_WEIGHTS)))
                    service = str(rng.choice(SERVICE_NAMES, p=SERVICE_WEIGHTS))
                    duration = SERVICES[service][1]
                    if cursor + timedelta(minutes=duration) > close:
                        break
                    pool = regulars if rng.random() < 0.85 else all_customer_ids
                    customer_id = pool[rng.integers(len(pool))]
                    booked_at = cursor - timedelta(minutes=int(rng.integers(60, 21 * 24 * 60)))
                    add_appointment(
                        rng, appts, shop, barber["barber_id"], customer_id, service,
                        cursor.astimezone(UTC), booked_at.astimezone(UTC), now, price_mult[shop["shop_id"]],
                    )
                    cursor += timedelta(minutes=duration)
        day += timedelta(days=1)
    return appts


def assign_customer_signups(
    rng: np.random.Generator, customers: list[dict], appts: list[dict], start: date, now: datetime
) -> None:
    first_booking: dict[str, datetime] = {}
    for a in appts:
        cid = a["customer_id"]
        if cid not in first_booking or a["booked_at"] < first_booking[cid]:
            first_booking[cid] = a["booked_at"]
    earliest = datetime.combine(start - timedelta(days=365), time(0), UTC)
    for c in customers:
        if c["customer_id"] in first_booking:
            lead = timedelta(days=int(rng.integers(0, 60)), minutes=int(rng.integers(0, 1440)))
            c["created_at"] = first_booking[c["customer_id"]] - lead
        else:
            c["created_at"] = earliest + (now - earliest) * rng.uniform()


def build_billing_events(
    rng: np.random.Generator, shops: list[dict], appts: list[dict], start: date, now: datetime
) -> list[dict]:
    events: list[dict] = []

    def emit(event_type, created, shop_id, data, amount=None, customer_id=None, appointment_id=None):
        if created > now:
            return
        events.append(
            {
                "event_id": make_id(rng, "evt", 12),
                "event_type": event_type,
                "created": created,
                "received_at": created + timedelta(seconds=int(rng.integers(0, 30))),
                "shop_id": shop_id,
                "customer_id": customer_id,
                "appointment_id": appointment_id,
                "amount_cents": amount,
                "currency": "usd",
                "data": data,
            }
        )

    # Shop SaaS subscriptions: signup, monthly renewals, upgrades, dunning on failed renewals.
    window_start = datetime.combine(start, time(0), UTC)
    for shop in shops:
        sid = shop["shop_id"]
        opened = datetime.combine(shop["opened_at"], time(12), UTC)
        if opened < window_start:
            signup = max(opened, window_start - timedelta(days=int(rng.integers(0, 400))))
        else:
            signup = opened - timedelta(days=int(rng.integers(0, 14)))
        signup += timedelta(minutes=int(rng.integers(0, 1440)))

        final_tier = shop["plan_tier"]
        tier, upgrade_at = final_tier, None
        if final_tier != "basic" and rng.random() < 0.4:
            tier = PLAN_TIERS[PLAN_TIERS.index(final_tier) - 1]
            upgrade_at = signup + (now - signup) * rng.uniform(0.2, 0.8)
        emit("subscription.created", signup, sid, {"object": "subscription", "plan": tier}, PLAN_PRICES[tier])

        period = 1
        while (renew_at := signup + relativedelta(months=period)) <= now:
            if upgrade_at and upgrade_at < renew_at:
                emit("subscription.upgraded", upgrade_at, sid,
                     {"object": "subscription", "plan": final_tier, "previous_plan": tier})
                tier, upgrade_at = final_tier, None
            amount = PLAN_PRICES[tier]
            if rng.random() >= SUBSCRIPTION_FAILURE_RATE:
                emit("subscription.renewed", renew_at, sid, {"object": "subscription", "plan": tier}, amount)
            else:
                recovered = False
                for attempt in range(1, 4):
                    emit("charge.failed", renew_at + timedelta(days=3 * (attempt - 1)), sid,
                         {"object": "subscription", "plan": tier,
                          "failure_code": str(rng.choice(FAILURE_CODES)), "retry_attempt": attempt}, amount)
                    if rng.random() < 0.6:
                        emit("subscription.renewed", renew_at + timedelta(days=3 * attempt), sid,
                             {"object": "subscription", "plan": tier, "retry_attempt": attempt}, amount)
                        recovered = True
                        break
                if not recovered:
                    emit("subscription.cancelled", renew_at + timedelta(days=9, hours=1), sid,
                         {"object": "subscription", "plan": tier, "cancel_reason": "payment_failed"})
                    break
            period += 1

    # Customer card/app charges for appointments (including no-show fees), plus refunds.
    for a in appts:
        if a["payment_method"] not in ("card", "app") or not a["price_cents"]:
            continue
        refs = {"customer_id": a["customer_id"], "appointment_id": a["appointment_id"]}
        amount = a["price_cents"] + (a["tip_cents"] or 0)
        charged_at = a["updated_at"] + timedelta(seconds=int(rng.integers(1, 120)))
        attempt = 1
        if rng.random() < CHARGE_FAILURE_RATE:
            emit("charge.failed", charged_at, a["shop_id"],
                 {"object": "charge", "charge_id": make_id(rng, "ch"), "payment_method": a["payment_method"],
                  "failure_code": str(rng.choice(FAILURE_CODES)), "retry_attempt": 1}, amount, **refs)
            charged_at += timedelta(minutes=int(rng.integers(1, 10)))
            attempt = 2
        charge_id = make_id(rng, "ch")
        emit("charge.succeeded", charged_at, a["shop_id"],
             {"object": "charge", "charge_id": charge_id, "payment_method": a["payment_method"],
              "retry_attempt": attempt}, amount, **refs)
        if a["status"] == "completed" and rng.random() < REFUND_RATE:
            full = rng.random() < 0.6
            refund = amount if full else int(round(amount * rng.uniform(0.2, 0.6) / 100) * 100)
            refunded_at = charged_at + timedelta(days=int(rng.integers(1, 11)), minutes=int(rng.integers(0, 1440)))
            emit("charge.refunded", refunded_at, a["shop_id"],
                 {"object": "refund", "charge_id": charge_id,
                  "refund_reason": str(rng.choice(REFUND_REASONS)), "partial": not full}, refund, **refs)

    events.sort(key=lambda e: (e["created"], e["event_id"]))
    return events


def inject_near_dupe_customers(
    rng: np.random.Generator, customers: list[dict], appts: list[dict], now: datetime, rate: float = 0.03
) -> dict[str, str]:
    """Same person signs up twice with a different id, email formatting and phone formatting.

    Some of their later appointments move to the new account, so dedupe matters downstream.
    Returns {dupe customer_id: original customer_id}.
    """
    appts_by_customer = defaultdict(list)
    for a in appts:
        appts_by_customer[a["customer_id"]].append(a)
    candidates = [c for c in customers if c["customer_id"] in appts_by_customer]
    picks = rng.choice(len(candidates), size=int(len(candidates) * rate), replace=False)

    dupe_of = {}
    next_id = len(customers) + 1
    for idx in sorted(picks):
        orig = candidates[idx]
        dupe = dict(orig)
        dupe["customer_id"] = f"cus_{next_id:06d}"
        next_id += 1
        dupe["email"] = vary_email(rng, orig["email"])
        dupe["phone"] = reformat_phone(rng, orig["phone"])
        if rng.random() < 0.3:
            dupe["last_name"] = orig["last_name"].upper()
        dupe["created_at"] = orig["created_at"] + (now - orig["created_at"]) * rng.uniform(0.1, 0.9)
        for a in appts_by_customer[orig["customer_id"]]:
            if a["booked_at"] >= dupe["created_at"] and rng.random() < 0.6:
                a["customer_id"] = dupe["customer_id"]
        customers.append(dupe)
        dupe_of[dupe["customer_id"]] = orig["customer_id"]
    return dupe_of


def crm_email(rng: np.random.Generator, fake: Faker, first: str, last: str, email: str) -> tuple[str | None, str]:
    """The CRM's copy of a booking email: lowercased and trimmed, and often not the same address."""
    email = email.strip().lower()
    local, domain = email.split("@")
    x = rng.random()
    if x < 0.60:
        return email, "email_exact"
    if x < 0.70:  # work email, or moved to another provider
        if rng.random() < 0.5:
            return email_for(first, last, fake.domain_name()), "email_different"
        return email_for(first, "", fake.free_email_domain(), str(int(rng.integers(10, 1000)))), "email_different"
    if x < 0.77:
        typos = EMAIL_DOMAIN_TYPOS.get(domain, [domain.replace(".com", ".con")])
        return f"{local}@{typos[rng.integers(len(typos))]}", "email_typo"
    if x < 0.84:  # gmail-style aliases that deliver to the same inbox
        if "+" in local:
            local = local.split("+")[0]
        elif "." in local:
            local = local.replace(".", "")
        else:
            local = f"{local}+booking"
        return f"{local}@{domain}", "email_variant"
    if x < 0.90:
        handle = re.sub(r"[^a-z]", "", f"{first[0]}{last}".lower())
        return f"{handle}{int(rng.integers(1, 100))}@{domain}", "email_local_differs"
    return None, "email_null"


def crm_phone(rng: np.random.Generator, phone: str) -> tuple[str | None, str]:
    """The CRM's copy of a booking phone: E.164 with extensions dropped, and often not the same number."""
    digits = phone_digits(phone)
    x = rng.random()
    if x < 0.65:
        return f"+1{digits}", "phone_exact"
    if x < 0.75:
        new = "".join(str(d) for d in rng.integers(0, 10, size=7))
        return f"+1{digits[:3]}{new}", "phone_different"
    if x < 0.80:
        d = list(digits)
        i = int(rng.integers(3, 9))
        if d[i] != d[i + 1] and rng.random() < 0.5:
            d[i], d[i + 1] = d[i + 1], d[i]
        else:
            d[i] = str((int(d[i]) + int(rng.integers(1, 10))) % 10)
        return f"+1{''.join(d)}", "phone_typo"
    if x < 0.85:
        return (digits if rng.random() < 0.5 else f"1{digits}"), "phone_format"
    return None, "phone_null"


def crm_name(rng: np.random.Generator, first: str, last: str) -> tuple[str, str, str]:
    x = rng.random()
    if x < 0.08 and first in NICKNAMES:
        return NICKNAMES[first], last, "name_nickname"
    if 0.08 <= x < 0.11:
        return last, first, "name_swapped"
    if 0.11 <= x < 0.16:
        return (first, last.upper(), "name_noise") if rng.random() < 0.5 else (f"{first} ", last, "name_noise")
    return first, last, "name_exact"


def build_crm_contacts(
    fake: Faker,
    rng: np.random.Generator,
    customers: list[dict],
    dupe_of: dict[str, str],
    shops: list[dict],
    start: date,
    now: datetime,
    report: dict,
) -> tuple[list[dict], list[dict]]:
    """Marketing CRM contacts for (most of) the same people, plus CRM-only leads.

    Returns (contacts, answer_key) where answer_key rows map crm_contact_id -> customer_id.
    """
    shop_by_id = {s["shop_id"]: s for s in shops}
    dupes_by_orig = defaultdict(list)
    for dupe_id, orig_id in dupe_of.items():
        dupes_by_orig[orig_id].append(dupe_id)
    all_customer_ids = [c["customer_id"] for c in customers]
    contacts: list[dict] = []
    answer_key: list[dict] = []
    tags = Counter()

    def location_for(shop):
        return shop["name"] if rng.random() < 0.6 else shop["city"]

    def activity_after(created):
        return created + (now - created) * rng.uniform()

    for c in customers:
        if c["customer_id"] in dupe_of:
            continue  # the CRM only ever saw one of the person's booking accounts
        if rng.random() < 0.10:
            tags["booking_customer_missing_from_crm"] += 1 + len(dupes_by_orig[c["customer_id"]])
            continue

        email, email_tag = crm_email(rng, fake, c["first_name"], c["last_name"], c["email"])
        phone, phone_tag = crm_phone(rng, c["phone"])
        first, last, name_tag = crm_name(rng, c["first_name"], c["last_name"])
        mismatch = [t for t in (email_tag, phone_tag, name_tag) if not t.endswith("_exact")]

        r = rng.random()
        if r < 0.03:
            # Points at the person's other booking account if they have one, else at a stranger.
            linked_dupes = dupes_by_orig[c["customer_id"]]
            if linked_dupes:
                external_id = linked_dupes[0]
                mismatch.append("external_id_dupe_account")
            else:
                external_id = all_customer_ids[rng.integers(len(all_customer_ids))]
                mismatch.append("external_id_wrong")
        elif r < 0.53:
            external_id = c["customer_id"]
        else:
            external_id = None

        created = min(c["created_at"] + timedelta(hours=int(rng.integers(0, 24 * 30))), now)
        opt_out = not c["marketing_opt_in"]
        if rng.random() < 0.10:
            opt_out = not opt_out
        contact = {
            "crm_contact_id": make_id(rng, "crm", 8),
            "booking_customer_id": external_id,
            "first_name": first,
            "last_name": last,
            "email": email,
            "phone": phone,
            "preferred_location": location_for(shop_by_id[c["home_shop_id"]]),
            "lifecycle_stage": "customer" if rng.random() < 0.8 else "lapsed",
            "lead_source": str(rng.choice(LEAD_SOURCES)),
            "email_opt_out": opt_out,
            "created_at": created,
            "last_activity_at": activity_after(created),
        }
        contacts.append(contact)
        tags.update(mismatch)
        match_type = "+".join(mismatch) or "clean"
        for customer_id in [c["customer_id"], *dupes_by_orig[c["customer_id"]]]:
            answer_key.append({"crm_contact_id": contact["crm_contact_id"], "customer_id": customer_id, "match_type": match_type})
    tags["linked_contacts"] = len(contacts)

    # CRM-internal duplicates: a list import created the same person again.
    linked = list(contacts)
    keyed = defaultdict(list)
    for row in answer_key:
        keyed[row["crm_contact_id"]].append(row)
    for idx in rng.choice(len(linked), size=int(len(linked) * 0.02), replace=False):
        orig = linked[idx]
        dupe = dict(orig)
        dupe["crm_contact_id"] = make_id(rng, "crm", 8)
        dupe["booking_customer_id"] = None
        dupe["phone"] = None if rng.random() < 0.5 else orig["phone"]
        dupe["lead_source"] = "import"
        dupe["created_at"] = activity_after(orig["created_at"])
        dupe["last_activity_at"] = activity_after(dupe["created_at"])
        contacts.append(dupe)
        for row in keyed[orig["crm_contact_id"]]:
            answer_key.append(row | {"crm_contact_id": dupe["crm_contact_id"], "match_type": row["match_type"] + "+crm_duplicate"})
    tags["crm_internal_dupes"] = int(len(linked) * 0.02)

    # CRM-only leads who never booked; some share a surname and shop with a real customer.
    window_start = datetime.combine(start, time(0), UTC)
    n_leads = int(len(customers) * 0.15)
    for _ in range(n_leads):
        first = fake.first_name()
        if rng.random() < 0.3:
            lookalike = customers[rng.integers(len(customers))]
            last, shop = lookalike["last_name"], shop_by_id[lookalike["home_shop_id"]]
        else:
            last, shop = fake.last_name(), shops[rng.integers(len(shops))]
        created = window_start + (now - window_start) * rng.uniform()
        contacts.append(
            {
                "crm_contact_id": make_id(rng, "crm", 8),
                "booking_customer_id": None,
                "first_name": first,
                "last_name": last,
                "email": email_for(first, last, fake.free_email_domain()) if rng.random() < 0.9 else None,
                "phone": f"+1{phone_digits(fake.phone_number())}" if rng.random() < 0.6 else None,
                "preferred_location": location_for(shop),
                "lifecycle_stage": "lead",
                "lead_source": str(rng.choice(LEAD_SOURCES)),
                "email_opt_out": bool(rng.random() < 0.3),
                "created_at": created,
                "last_activity_at": activity_after(created),
            }
        )
    tags["crm_only_leads"] = n_leads

    report.update(sorted(tags.items()))
    contacts.sort(key=lambda r: r["crm_contact_id"])
    answer_key.sort(key=lambda r: (r["crm_contact_id"], r["customer_id"]))
    return contacts, answer_key


# --------------------------------------------------------------------------- messes


def inject_exact_dupes(df: pd.DataFrame, rng: np.random.Generator, rate: float) -> tuple[pd.DataFrame, int]:
    n = int(len(df) * rate)
    dupes = df.iloc[rng.choice(len(df), size=n, replace=False)]
    return pd.concat([df, dupes]).sort_index(kind="stable").reset_index(drop=True), n


def mess_shops(df: pd.DataFrame, rng: np.random.Generator, report: dict) -> pd.DataFrame:
    # Hand-entered in the admin tool, so every row picked its own date format.
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y"]
    picks = rng.integers(len(formats), size=len(df))
    df["opened_at"] = [d.strftime(formats[i]) for d, i in zip(df["opened_at"], picks)]
    report["opened_at_non_iso"] = int((picks != 0).sum())

    zip4 = rng.random(len(df)) < 0.25
    df.loc[zip4, "zipcode"] = [f"{z}-{int(rng.integers(0, 10000)):04d}" for z in df.loc[zip4, "zipcode"]]
    report["zip_plus4"] = int(zip4.sum())
    return df


def mess_barbers(df: pd.DataFrame, rng: np.random.Generator, report: dict) -> pd.DataFrame:
    df["hired_at"] = [d.isoformat() for d in df["hired_at"]]
    active = df["terminated_at"].isna()
    # Active barbers: legit null, but the roster export spells "not terminated" three ways.
    tokens = [[None, "NULL", "0000-00-00"][i] for i in rng.choice(3, size=int(active.sum()), p=[0.7, 0.2, 0.1])]
    df["terminated_at"] = [d.isoformat() if d else None for d in df["terminated_at"]]
    df.loc[active, "terminated_at"] = tokens
    report["terminated_at_null_variants"] = sum(t is not None for t in tokens)

    no_email = rng.random(len(df)) < 0.05
    df.loc[no_email, "email"] = None
    report["null_email"] = int(no_email.sum())

    df["chair_rent_cents"] = df["chair_rent_cents"].astype("Int64")
    return df


def mess_customers(df: pd.DataFrame, rng: np.random.Generator, report: dict) -> pd.DataFrame:
    df["created_at"] = [iso_utc(d) for d in df["created_at"]]
    for col, rate in [("email", 0.03), ("phone", 0.05)]:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = null_tokens(rng, int(mask.sum()))
        report[f"null_{col}"] = int(mask.sum())
    df, report["exact_dupes"] = inject_exact_dupes(df, rng, 0.015)
    return df


def mess_appointments(
    df: pd.DataFrame, rng: np.random.Generator, shops: list[dict], report: dict
) -> pd.DataFrame:
    # Stale versions: the export captured an earlier "booked" snapshot of finished appointments.
    finished = df[df["status"].isin(["completed", "no_show", "cancelled"])]
    stale = finished.sample(n=int(len(df) * 0.02), random_state=rng).copy()
    stale["status"] = "booked"
    stale["updated_at"] = stale["booked_at"]
    stale["tip_cents"] = None
    stale["payment_method"] = None
    df = pd.concat([df, stale]).sort_index(kind="stable").reset_index(drop=True)
    report["stale_versions"] = len(stale)

    df, report["exact_dupes"] = inject_exact_dupes(df, rng, 0.01)

    unassigned = rng.random(len(df)) < 0.02
    df.loc[unassigned, "barber_id"] = None
    report["null_barber_id"] = int(unassigned.sum())

    walk_ins = rng.random(len(df)) < 0.01
    df.loc[walk_ins, "customer_id"] = None
    report["null_customer_id"] = int(walk_ins.sum())

    orphans = (rng.random(len(df)) < 0.005) & ~walk_ins
    df.loc[orphans, "customer_id"] = [f"cus_9{int(rng.integers(0, 100000)):05d}" for _ in range(orphans.sum())]
    report["orphan_customer_id"] = int(orphans.sum())

    recased = 0
    for status, variants in STATUS_VARIANTS.items():
        mask = (df["status"] == status) & (rng.random(len(df)) < 0.03)
        df.loc[mask, "status"] = [variants[i] for i in rng.integers(len(variants), size=int(mask.sum()))]
        recased += int(mask.sum())
    report["status_case_variants"] = recased

    df["tip_cents"] = df["tip_cents"].astype("Int64")
    df["price_cents"] = df["price_cents"].astype("Int64").astype(object)
    dollars = (rng.random(len(df)) < 0.02) & df["price_cents"].notna().to_numpy()
    df.loc[dollars, "price_cents"] = [f"{p / 100:.2f}" for p in df.loc[dollars, "price_cents"]]
    report["price_as_dollar_string"] = int(dollars.sum())

    # Timezones: most rows are UTC ISO-8601, but one legacy shop (and a slice of everyone
    # else) writes naive shop-local time, some write offsets, and a few use US format.
    shop_tz = {s["shop_id"]: ZoneInfo(s["timezone"]) for s in shops}
    legacy_shop = shops[rng.integers(len(shops))]["shop_id"]
    styles = Counter()
    formatted = []
    for start, shop_id, x in zip(df["scheduled_start"], df["shop_id"], rng.random(len(df))):
        local = start.astimezone(shop_tz[shop_id])
        if shop_id == legacy_shop or 0.80 <= x < 0.90:
            style, value = "naive_local", local.strftime("%Y-%m-%d %H:%M:%S")
        elif x < 0.80:
            style, value = "utc_z", iso_utc(start)
        elif x < 0.97:
            style, value = "local_with_offset", local.isoformat(timespec="seconds")
        else:
            style, value = "us_format_local", local.strftime("%m/%d/%Y %I:%M %p")
        styles[style] += 1
        formatted.append(value)
    df["scheduled_start"] = formatted
    report["legacy_naive_local_shop"] = legacy_shop
    report.update({f"scheduled_start_{k}": v for k, v in sorted(styles.items())})

    df["booked_at"] = [iso_utc(d) for d in df["booked_at"]]
    # Legacy app server stamps updated_at in its own naive Central time.
    df["updated_at"] = [d.astimezone(LEGACY_SERVER_TZ).strftime("%Y-%m-%d %H:%M:%S") for d in df["updated_at"]]
    return df


def mess_crm_contacts(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at"] = [d.astimezone(CRM_PORTAL_TZ).strftime("%Y-%m-%d %H:%M:%S") for d in df["created_at"]]
    df["last_activity_at"] = [int(d.timestamp() * 1000) for d in df["last_activity_at"]]
    return df


def mess_billing_events(events: list[dict], rng: np.random.Generator, report: dict) -> list[dict]:
    # Webhook replays: the processor redelivers the same event_id later.
    replays = []
    for i in rng.choice(len(events), size=int(len(events) * 0.02), replace=False):
        replay = dict(events[i])
        replay["received_at"] = replay["received_at"] + timedelta(seconds=int(rng.integers(60, 6 * 3600)))
        replays.append(replay)
    events = events + replays
    report["webhook_replays"] = len(replays)

    ms = rng.random(len(events)) < 0.03
    upper = rng.random(len(events)) < 0.05
    drop_customer = rng.random(len(events)) < 0.01
    out = []
    for e, is_ms, is_upper, is_dropped in zip(events, ms, upper, drop_customer):
        e = dict(e)
        epoch = int(e["created"].timestamp())
        e["created"] = epoch * 1000 if is_ms else epoch
        if is_upper:
            e["currency"] = "USD"
        if is_dropped and e["customer_id"]:
            e["customer_id"] = None
            report["null_customer_id"] = report.get("null_customer_id", 0) + 1
        out.append(e)
    report["created_epoch_millis"] = int(ms.sum())
    report["currency_uppercase"] = int(upper.sum())

    # The log is in delivery order, so replays land after the original.
    out.sort(key=lambda e: (e["received_at"], e["event_id"]))
    for e in out:
        e["received_at"] = iso_utc(e["received_at"])
    return out


# --------------------------------------------------------------------------- output


def write_outputs(
    out_dir: Path, frames: dict[str, pd.DataFrame], events: list[dict], answer_key: list[dict]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    with open(out_dir / "billing_events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    # Kept outside the raw landing zone so models can't read the answers by accident.
    key_dir = out_dir.parent / "answer_key"
    key_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(answer_key).to_csv(key_dir / "crm_customer_matches.csv", index=False)
    return key_dir


def print_summary(out_dir: Path, start: date, end: date, seed: int, rows: dict, report: dict) -> None:
    print(f"Wrote {out_dir}/ (seed={seed}, window {start} .. {end})")
    for source, n in rows.items():
        print(f"\n  {source}: {n:,} rows")
        for mess, count in report.get(source, {}).items():
            print(f"    {mess:<36} {count:,}" if isinstance(count, int) else f"    {mess:<36} {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--shops", type=int, default=12)
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--days", type=int, default=270)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Faker.seed(args.seed)
    fake = Faker("en_US")
    rng = np.random.default_rng(args.seed)

    end = END_DATE
    start = end - timedelta(days=args.days - 1)
    now = datetime.combine(end + timedelta(days=1), time(0), UTC)
    report: dict[str, dict] = defaultdict(dict)

    shops = build_shops(fake, rng, args.shops, start, end)
    barbers = build_barbers(fake, rng, shops, end)
    customers = build_customers(fake, rng, args.customers, shops)
    appts = build_appointments(rng, shops, barbers, customers, start, end, now)
    assign_customer_signups(rng, customers, appts, start, now)
    dupe_of = inject_near_dupe_customers(rng, customers, appts, now)
    report["customers"]["near_dupe_customers"] = len(dupe_of)
    events = build_billing_events(rng, shops, appts, start, now)

    # The CRM gets its own random streams so adding it doesn't shift the other sources.
    crm_fake = Faker("en_US")
    crm_fake.seed_instance(args.seed + 1)
    crm_contacts, answer_key = build_crm_contacts(
        crm_fake, np.random.default_rng([args.seed, 1]), customers, dupe_of, shops, start, now, report["crm_contacts"]
    )

    frames = {
        "shops": mess_shops(pd.DataFrame(shops), rng, report["shops"]),
        "barbers": mess_barbers(pd.DataFrame(barbers), rng, report["barbers"]),
        "customers": mess_customers(pd.DataFrame(customers), rng, report["customers"]),
        "appointments": mess_appointments(pd.DataFrame(appts), rng, shops, report["appointments"]),
    }
    events = mess_billing_events(events, rng, report["billing_events"])
    frames["crm_contacts"] = mess_crm_contacts(pd.DataFrame(crm_contacts))

    key_dir = write_outputs(args.out, frames, events, answer_key)
    rows = {name: len(df) for name, df in frames.items()} | {"billing_events": len(events)}
    print_summary(args.out, start, end, args.seed, rows, report)
    print(f"\nAnswer key: {key_dir}/crm_customer_matches.csv ({len(answer_key):,} rows)")


if __name__ == "__main__":
    main()
