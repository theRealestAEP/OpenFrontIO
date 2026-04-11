# Oil Rig System

This document describes the current implemented oil-rig system in OpenFrontIO.

## Summary

- `Oil Rig` is a buildable, upgradable structure.
- `Oil Rig` still uses one build action, but can now target both land oil and offshore ocean oil.
- Oil fields are deterministic, public map features generated from the map/config seed.
- Oil reserves are finite.
- land rigs pay directly, while offshore rigs deliver oil income by cargo ship.
- Oil rigs behave like special train stops for rail interactions.
- Oil fields visually fade as they deplete.

I have seen probably 5/6 requests for Oil as a feature for the game but thought the ideas were missing pieces, primarily why you would want it, especially when factories already serve as a de-facto 'infinite' internal gold/resource. But I really liked the idea of having simple resources play a factor in spawn location choices, impact mid/late game economics, add in another layer of snowballing or snowball prevention.

So this is kind of what I cooked up mentally:

I think the best way to go about it is to have a new map layer for oil as a resource thats procedurally created each game. This changes map dynamics every game slightly and your spawn is no longer just a matter of terrain maxing but also resource maxing.

Oil should be a finite resource thats not easily accessible early game, but also does not perpetually stack/blow the economy in the late games. It should serve as a mid-game economic spurt that can generate a lot of money very quickly but for a short period of time. This will cause earlier bombs, and early/quick industrialization that is separate to raw domination or early trade. They also should create a new type of economic competition between you and neighbors, for example: You and a neighbor are allies but share a single oil reserve, how many oil rigs will you build before it becomes economically detrimental to build more? is it worth building more just so your neighbor has less access to that resource?

The returns on oil rigs reserve should be logarithmic.

Oil rigs should network up like cities and factories while they are active. Once an oil reserve is depleted the Rig is defunct. No more cash no more trains.

Cash flow from oil rigs should be instant and sustained. Its a massive economic advantage especially for isolated players who may not otherwise have trade infrastructure.

## TL:DR

The oil-rig system is meant to add a temporary but high-impact economic layer to the map.

The main purpose of the feature is:

- to create real resource pressure/contention around specific land
- to make the mid game more explosive by giving players a way to spike economically without relying only on conquest tempo or long-lived rail scaling
- to make land that overlaps with oil meaningfully more valuable than generic territory
- to create interesting decisions around when to spread out, when to stack, and when to race an opponent for a field before it burns down
- to make depleted economic structures feel different from permanent infrastructure

In practice, this means oil is designed to be:

- strong when first secured
- contestable while active
- exhaustible over time
- much less relevant once the reserves are gone

That combination is important for pacing. Oil can accelerate a player's mid-game power and reward aggressive land grabs, but because reserves are finite, it does not permanently dominate the late game the way an infinite compounding income source would.

ANYWAYS

I realize this was something thats hard to describe without showing/making it so I actually made a quick fork of the game and put together a rough but functional version of this building type!

(I'm not a code contributor and this is my first post entirely so this fork really is only for demo purposes not prod code)
(im also not an artist I made the assets in like 2 seconds)

Attached is a video of a playthrough of me and a bunch of nations building and using oil rigs. This has not been balanced at all or tested in live games.

## Current behavior:

- 1st oil rig: `500,000`
- 2nd+ oil rig: `1,000,000`
- land clicks keep the original land-rig flow
- ocean clicks launch an `OilRigShip` from the best owned reachable port and convert into an `Oil Rig` on arrival
- offshore rigs share the same reserve pool, depletion rules, dormancy, upgrade path, and pricing as land rigs
- offshore rigs contribute `1.5x` extraction weight relative to an equivalent land rig
- land-rig gold still pays directly
- offshore rigs buffer extracted gold and launch a `TradeShip` cargo run every `10s`
- offshore rigs become inert if they do not have a reachable owned active port
- finished offshore rigs are captured by `TransportShip` boarding, not warships

- oil reserve sizes:
  - small: `8,000`
  - medium: `16,000`
  - large: `24,000`

Because reserves are finite, each field has a fixed total gross value:

- small field: `8,000 * 1,200 = 9.6M`
- medium field: `16,000 * 1,200 = 19.2M`
- large field: `24,000 * 1,200 = 28.8M`

I have been testing with 24 - 160 oil fields so 230.4M to 4608M in extra cash on the board.

Depletion rate: 30 _ ln(1 + effectiveRigCount) oil units per second
Cash gen rate: gold/sec = 36,000 _ ln(1 + effectiveRigCount)

1 effective rig: 20.8 oil/sec ~24.95k/sec
2: 33.0 oil/sec ~49.91k/sec
3: 41.6 oil/sec ~49.91k/sec
5: 53.8 oil/sec ~64.50k/sec
10: 71.9 oil/sec ~86.32k/sec

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

- field centers still come from land candidates
- field footprints now include both land tiles and ocean tiles inside the field radius
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

- Land placement:
  - the tile is owned by the player
  - the tile is valid for normal structure placement/spacing
  - the tile sits on at least one oil field with remaining reserve
- Offshore placement:
  - the clicked tile is ocean
  - the tile sits on at least one oil field with remaining reserve
  - the tile is valid under the usual oil-rig spacing rules
  - the player owns a reachable port on the same water component

Blocked cases:

- unowned land
- ocean oil without a reachable owned port
- normal invalid structure placement
- tiles with no oil
- tiles where every overlapping field has already been exhausted

## Offshore Deployment

Ocean placement uses a port-launched deployment step instead of instant structure spawn.

Current offshore flow:

- clicking a valid ocean oil tile keeps the `Oil Rig` button unchanged for the user
- the game launches an internal `OilRigShip` from the best owned reachable port
- the deploy ship reuses the current trade-ship sprite in v1
- when the ship reaches the selected ocean oil tile, it is removed and replaced with a normal `Oil Rig`
- the final offshore rig then goes through the usual oil-rig construction duration
- if the destination becomes blocked or unreachable before arrival, the deploy ship is removed and the full rig cost is refunded
- if the field depletes during travel, the offshore rig still finishes construction and simply starts dormant

## Extraction Model

Oil extraction is grouped by:

- oil field
- owner

For each owner on a field:

- land rig weight = `rig.level()`
- offshore rig weight = `1.5 * rig.level()`
- `effectiveRigCount` is the weighted sum across that owner's active rigs on that field
- extraction per second is:

```text
30 * ln(1 + effectiveRigCount)
```

This means:

- one level-5 land rig on a field is equivalent to five level-1 land rigs on the same field
- one level-1 offshore rig counts like `1.5` level-1 land rigs for both drain speed and payout share
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

## Cash Behavior

- the total gold for that owner/field/second is computed first
- then it is split across the participating rigs proportionally by the same weighted contribution used for extraction
- land rigs receive their split immediately as direct gold
- offshore rigs buffer their split locally and launch it as cargo on a `TradeShip` every `100` ticks
- offshore cargo targets the nearest reachable owned active port at launch time
- offshore cargo reroutes to another owned active port if the original port becomes invalid mid-route
- if no owned active port is reachable, the buffered cargo stays on the rig and the rig remains inert until service returns

## Naval Capture

Finished offshore rigs stay as normal `Oil Rig` units on ocean tiles.

Current capture rules:

- hostile warships no longer capture finished offshore rigs directly
- hostile warships can still capture offshore oil cargo ships, because those use the normal `TradeShip` unit
- finished offshore rigs are treated like single-pixel islands and are captured by sending a `TransportShip` directly to the rig tile
- capture happens in place on arrival; the rig is not destroyed and the ocean tile itself is not conquered
- the deploy ship is not capturable in v1
- captured offshore rigs immediately begin yielding for the new owner while still draining the same shared field reserve

## Rail Behavior

Active oil rigs integrate with rail systems.

Current rail rules:

- active oil rigs are valid rail stop targets
- factories can connect to active oil rigs
- oil rigs do not spawn trains
- train stops at active oil rigs pay `2x` the normal city stop payout
- depleted/dormant rigs are not active rail stops
- offshore rigs do not get special new rail rules; they remain normal `Oil Rig` units and only gain rail value if the existing rail system can already connect them

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

Broader gameplay implications:

- oil creates pressure to fight for specific pieces of land, not just for border shape or city count
- players who temporarily slow down direct expansion can still create a sharp economic spike by securing and drilling oil
- overlapping oil territory becomes strategically important because the land itself may be worth contesting even before its normal infrastructure value is considered
- contested fields deplete faster when multiple owners drill them at once, which adds urgency and makes shared deposits naturally unstable
- once a well is empty, the rig remains but no longer contributes economically, which creates a real reason to clear or deconstruct it later if building space becomes tight
- because each field contains a fixed amount of total value, oil pushes timing, tempo, and map control much more than it pushes endless late-game snowballing

## Tests

Current targeted coverage includes:

- deterministic field generation
- buildability on owned active land tiles
- buildability on reachable offshore oil tiles
- blocked placement when no valid oil remains
- deploy-ship launch/conversion for offshore rigs
- weighted offshore extraction behavior
- offshore cargo launch, reroute, and deletion
- offshore inert behavior without a reachable port
- offshore capture by transport ships
- offshore cargo interception by warships
- passive land extraction and depletion
- overlap fall-through behavior via field lookup semantics
- rail integration tests for oil-rig stops

Relevant tests:

- `tests/core/game/OilField.test.ts`
- `tests/Warship.test.ts`
- `tests/core/game/TrainStation.test.ts`
- `tests/core/game/RailNetwork.test.ts`

## Key Files

- `src/core/game/OilField.ts`
- `src/core/execution/OilExecution.ts`
- `src/core/execution/OffshoreOilRigExecution.ts`
- `src/core/execution/OilRigExecution.ts`
- `src/core/execution/TradeShipExecution.ts`
- `src/core/execution/TransportShipExecution.ts`
- `src/core/execution/WarshipExecution.ts`
- `src/core/game/GameImpl.ts`
- `src/core/game/GameView.ts`
- `src/core/game/OilRigUtils.ts`
- `src/core/game/PlayerImpl.ts`
- `src/core/game/TrainStation.ts`
- `src/core/game/RailNetworkImpl.ts`
- `src/client/graphics/layers/TerrainLayer.ts`
- `src/client/graphics/layers/StructureLayer.ts`
- `src/client/graphics/layers/StructureDrawingUtils.ts`
