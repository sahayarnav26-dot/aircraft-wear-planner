# Synthetic supply-chain data

**Everything in this folder is synthetic demonstration data.** Station names and coordinates are
real Indian airports, but the network, stock, prices, lead times, suppliers and fleet are invented.
Regenerate with `python data/generate_synthetic.py` (fixed seed 42, so the files are identical each run).

| File | Columns | Notes |
| --- | --- | --- |
| parts.csv | part_id, part_name, component, unit_weight_kg, unit_cost_usd | FD001 wear (HPC degradation) maps to the HPC blade set. unit_cost_usd is the replacement value |
| locations.csv | location_id, name, type, lat, lon | Hub, line stations and a central warehouse |
| inventory.csv | location_id, part_id, qty_on_hand, qty_reserved, reorder_point | Six free HPC blade sets for 20 engines, so stock is scarce on purpose |
| transfers.csv | from_location, to_location, mode, transit_hours, cost_usd, distance_km | Every pair, both directions, road and air |
| suppliers.csv | supplier_id, supplier_name, part_id, origin_location, lead_time_days_standard, lead_time_days_expedited, unit_price_usd, expedite_fee_usd, standard_mode, distance_km | A domestic (road) and an international (sea) source per part; expedited is always air |
| fleet.csv | aircraft_id, engine_id, base_location, cycles_per_day, next_departure, current_cycle | 10 aircraft, 2 engines each; engine_id matches outputs/predictions.csv; current_cycle is the scenario start |

Emission factors are **not** synthetic: see `data/reference/emission_factors.csv` and its source column.
