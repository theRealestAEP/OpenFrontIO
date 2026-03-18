# Oil Rig System

This document describes the current implemented oil-rig system in OpenFrontIO.

## Summary

- `Oil Rig` is a buildable, upgradable structure.
- Oil rigs can only be built on owned land tiles that currently sit on at least one oil field with remaining reserve.
- Oil fields are deterministic, public map features generated from the map/config seed.
- Oil reserves are finite.
- Oil income is passive.
- Oil rigs behave like special train stops for rail interactions.
- Oil fields visually fade as they deplete.

## Unit Behavior

`Oil Rig` is registered as a structure-level `UnitType`.

Current behavior:

- base build cost starts at `500,000`
- upgradable
- construction duration matches cities/factories
- remains on the map after depletion
- becomes dormant when no overlapping oil field under its tile has reserve left

Dormant oil rigs:

- generate no oil income
- are not treated as active oil train stops
- render in a muted/inactive state

## Price Scaling

Oil rig pricing currently uses this curve:

- 1st oil rig: `500,000`
- 2nd oil rig: `1,000,000`
- 3rd+: `1,000,000`

This intentionally keeps the higher first-build price while capping like factories.

## Oil Field Generation

Oil fields are generated procedurally in shared game code.

Current field generation settings:

- min fields: `24`
- max fields: `160`
- target count: about `landTiles / 18,000`
- field radii:
  - small: `80`
  - medium: `112`
  - large: `144`
- reserve sizes:
  - small: `8,000`
  - medium: `16,000`
  - large: `24,000`

Generation notes:

- only land tiles qualify
- fields are deterministic from map identity/size
- the system is intentionally aggressive right now so large maps feel oil-rich
- fields may overlap

## Overlapping Fields

Tiles can belong to multiple oil fields.

Current overlap behavior:

- a rig on an overlapping tile draws from the first overlapping field with remaining reserve
- if that field depletes, the rig automatically falls through to the next overlapping field under the same tile
- the rig only goes dormant when every overlapping oil field under that tile is empty

This means overlapping reserves effectively behave like layered deposits.

## Placement Rules

An oil rig can be built only if all of these are true:

- the tile is owned by the player
- the tile is valid for normal structure placement/spacing
- the tile sits on at least one oil field with remaining reserve

Blocked cases:

- unowned land
- normal invalid structure placement
- tiles with no oil
- tiles where every overlapping field has already been exhausted

## Extraction Model

Oil extraction is grouped by:

- oil field
- owner

For each owner on a field:

- `effectiveRigCount = sum(rig.level())` across that owner's active rigs on that field
- extraction per second is:

```text
30 * ln(1 + effectiveRigCount)
```

This means:

- one level-5 rig on a field is equivalent to five level-1 rigs on the same field
- multiple rigs on the same field stack into the same diminishing-returns bucket
- spreading rigs across multiple fields is more efficient than concentrating them all on one field


Implementation details:

- game rate is effectively `10 ticks/sec`
- oil execution only pays on every 10th tick
- extraction is calculated in per-second chunks

This keeps the same economy rate while making payouts easier to read and reason about.

## Gold Conversion

Current gold conversion:

- `1,200 gold` per oil unit

Because reserves are finite, each field has a fixed total gross value:

- small field: `8,000 * 1,200 = 9.6M`
- medium field: `16,000 * 1,200 = 19.2M`
- large field: `24,000 * 1,200 = 28.8M`

More rigs do not increase total field value.
They only extract the same reserve faster.

## Cash  Behavior

- the total gold for that owner/field/second is computed first
- then it is split across the participating rigs proportionally by rig level

## Rail Behavior

Active oil rigs integrate with rail systems.

Current rail rules:

- active oil rigs are valid rail stop targets
- factories can connect to active oil rigs
- oil rigs do not spawn trains
- train stops at active oil rigs pay `2x` the normal city stop payout
- depleted/dormant rigs are not active rail stops

## Visuals and Discovery

Current oil visibility rules:

- pressing `Space` anywhere shows the oil overlay
- selecting `Oil Rig` for placement also shows oil automatically, even without `Space`

Current depletion visualization:

- field footprint does not visually shrink
- oil fields fade as reserves drop
- empty fields disappear from the oil overlay

This matches the current gameplay rule: depletion is tile/reserve based, not footprint-shrink based.

## Rendering / Assets

There are separate asset responsibilities for close-up and zoomed-out rendering.

### Close-up / stylized structure view

Used by:

- `src/client/graphics/layers/StructureLayer.ts`

Current oil-rig close-up asset:

- `resources/images/buildings/oilrig1.png`

### Zoomed-out icon view

Used by:

- `src/client/graphics/layers/StructureDrawingUtils.ts`
- `src/client/graphics/layers/StructureIconsLayer.ts`

The zoomed-out oil-rig icon currently uses the same `oilrig1.png` source image and is tinted like other player-owned structures.

### UI/build-menu icon

Used by:

- build menu
- help modal
- unit display
- player overlay

Current flat icon asset:

- `resources/images/OilRigIconWhite.svg`

## Strategic Implications

With the current formula:

- first rigs on fresh fields are very efficient
- extra rigs on the same field are mainly a speed/front-loading decision
- stacking and upgrading are economically equivalent on the same field if they produce the same total level sum
- spreading rigs across different fields is usually the strongest long-term play

Examples:

- 1 rig on 1 field is much more efficient than adding that same rig as the 5th rig on an already-developed field
- 5 rigs on the same field are equivalent to one level-5 rig on that field
- 5 rigs across 5 different fields are much stronger than 5 rigs concentrated on 1 field

## Tests

Current targeted coverage includes:

- deterministic field generation
- buildability on owned active field tiles
- blocked placement when no valid oil remains
- passive extraction and depletion
- overlap fall-through behavior via field lookup semantics
- rail integration tests for oil-rig stops

Relevant tests:

- `tests/core/game/OilField.test.ts`
- `tests/core/game/TrainStation.test.ts`
- `tests/core/game/RailNetwork.test.ts`

## Key Files

- `src/core/game/OilField.ts`
- `src/core/execution/OilExecution.ts`
- `src/core/execution/OilRigExecution.ts`
- `src/core/game/GameImpl.ts`
- `src/core/game/GameView.ts`
- `src/core/game/PlayerImpl.ts`
- `src/core/game/TrainStation.ts`
- `src/core/game/RailNetworkImpl.ts`
- `src/client/graphics/layers/TerrainLayer.ts`
- `src/client/graphics/layers/StructureLayer.ts`
- `src/client/graphics/layers/StructureDrawingUtils.ts`
