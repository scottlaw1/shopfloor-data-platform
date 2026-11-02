# shopfloor-data-platform

A learning data platform for a barbershop SaaS, built on DuckDB, dbt-duckdb, and Dagster.

## Seed data

```sh
uv run python generate_seed.py            # writes data/raw/, seed 42
uv run python generate_seed.py --seed 7 --shops 20 --customers 5000 --days 365
```

The output is deterministic for a given `--seed`. The window ends on a fixed date (2026-11-07), and the default 270 days cover both the March and November DST transitions.

| file | pretends to be |
| --- | --- |
| `shops.csv` | admin/CRM export |
| `barbers.csv` | staff roster export |
| `customers.csv` | booking-app signup export |
| `appointments.csv` | booking-system export (statuses: completed, no_show, cancelled, rescheduled, booked) |
| `billing_events.jsonl` | payment-processor webhook log (subscriptions, charges, failures, refunds) |
| `crm_contacts.csv` | marketing CRM export (its own contact IDs; emails/phones don't line up with booking) |

`data/answer_key/crm_customer_matches.csv` holds the true `crm_contact_id → customer_id` mapping, with a `match_type` that names the mismatches applied to each pair. Use it to score customer-matching logic, and never read it from models.

### Known messes (the cleanup checklist)

The generator plants these on purpose and prints a count for each one when it runs.

#### Duplicates

- `customers`: exact duplicate rows. Near-duplicates where the same person has a second `customer_id`, an email with different case, whitespace, or a `+tag`, a reformatted phone number, and sometimes an UPPERCASE last name. Some of their later appointments sit under the duplicate ID.
- `appointments`: exact duplicate rows. Stale snapshots that repeat an `appointment_id` with `status=booked` and an older `updated_at`, so the latest `updated_at` should win.
- `billing_events`: webhook replays that repeat an `event_id` with a later `received_at`.

#### Nulls and blanks

- `customers.email` and `customers.phone` show nulls four ways: empty, `NULL`, `N/A`, or `" "`.
- `barbers.terminated_at` is a legitimate null for active barbers, but the export spells it as empty, `NULL`, or `0000-00-00`. `barbers.email` is sometimes empty.
- `barbers.commission_pct` and `barbers.chair_rent_cents`: exactly one of the two is set, by design.
- `appointments`: `barber_id` is null on some rows (unassigned), `customer_id` is null on walk-ins, `customer_id` sometimes points to a customer that doesn't exist (`cus_9xxxxx`), and `tip_cents` is null on unrecorded cash tips and on future bookings.
- `billing_events.customer_id` is sometimes null on charge events.

#### Timezones

- `appointments.scheduled_start` comes in four forms:
  - mostly UTC `...Z`
  - naive shop-local time; one "legacy" shop (named in the run summary) uses this for every row
  - local time with an offset (`-05:00`)
  - US format `MM/DD/YYYY hh:mm AM`

  Use `shops.timezone` to resolve them. The Arizona shops never observe DST.
- `appointments.booked_at` is UTC, but `appointments.updated_at` is naive **America/Chicago** time, written by a legacy server.
- `billing_events.created` is usually epoch seconds and sometimes epoch milliseconds. `received_at` is ISO UTC.
- `shops.opened_at` mixes four date formats (`2022-02-11`, `02/11/2022`, `February 11, 2022`, `11-Feb-2022`).

#### Format noise

- `appointments.status` has casing and spelling variants (`No-Show`, `NOSHOW`, `COMPLETED`, `canceled`, and others).
- `appointments.price_cents` sometimes holds a dollar string (`"35.00"`).
- `billing_events.currency` is mostly `usd` and sometimes `USD`.
- `shops.zipcode` is sometimes ZIP+4, and some ZIPs have leading zeros, so read it as a string.
- Phone numbers use whatever formats Faker produced (extensions, `001-` prefixes, dots, and so on).

#### CRM vs booking (entity resolution)

- `crm_contacts.booking_customer_id` is filled on only about half of linked contacts, and about 3% of those point at the wrong customer. You have to match on attributes.
- About 10% of booking customers never made it into the CRM. About 15% extra CRM contacts are leads who never booked. Some leads share a surname and shop with a real customer.
- A booking customer's two near-duplicate accounts map to **one** CRM contact. About 2% of CRM contacts are internal duplicates (`lead_source=import`).
- The CRM stores emails lowercase and trimmed. About 40% still differ from booking:
  - a different address (work email or new provider)
  - a domain typo (`gmial.com`, `hotmail.con`)
  - a Gmail alias (dots removed, `+tag` added or stripped)
  - a different local part (`jsmith42@`)
  - null
- The CRM stores phones as E.164 (`+17125551234`) with no extensions. About 35% still differ from booking:
  - a different number
  - a one-digit typo or two transposed digits
  - 10 or 11 digits with no `+`
  - null
- Names: nicknames (William ↔ Bill), swapped first and last names, an UPPERCASE last name, trailing spaces.
- `email_opt_out` has the opposite meaning of booking's `marketing_opt_in`, and about 10% of contacts genuinely disagree.
- `preferred_location` is free text (shop name or city), not a `shop_id`.
- Timestamps: `created_at` is naive **America/New_York** time (the CRM portal's timezone). `last_activity_at` is epoch milliseconds.

With seed 42, a naive email join finds about 57% of the true pairs and a phone-digits join finds about 68%. Using both finds about 86%, plus 16 false positives.
