# Tear Sheets

Tear sheets are serializable report data models. They are separate from the
renderer so reports can be generated, stored, tested, and rendered later.

```python
import ferric_alpha as fa

summary = fa.tears.create_summary_tear_sheet_data(factor_data)
returns = fa.tears.create_returns_tear_sheet_data(factor_data)
information = fa.tears.create_information_tear_sheet_data(factor_data)
turnover = fa.tears.create_turnover_tear_sheet_data(factor_data)
full = fa.tears.create_full_tear_sheet_data(factor_data)
```

## Serialization

```python
payload = full.to_json()
round_tripped = fa.tears.TearSheetData.from_json(payload)
```

The tear-sheet contract is deterministic: tables have stable identifiers,
typed columns, explicit nulls, and sorted rows.
