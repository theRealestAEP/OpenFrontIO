import { Config } from "../configuration/Config";
import { PseudoRandom } from "../PseudoRandom";
import { simpleHash } from "../Util";
import { GameMap, TileRef } from "./GameMap";

const OIL_FIELD_MIN_COUNT = 24;
const OIL_FIELD_MAX_COUNT = 160;
const OIL_FIELD_MIN_RESERVE = 8_000;
const OIL_FIELD_MEDIUM_RESERVE = 16_000;
const OIL_FIELD_LARGE_RESERVE = 24_000;
const OIL_FIELD_MIN_RADIUS = 80;
const OIL_FIELD_MEDIUM_RADIUS = 112;
const OIL_FIELD_LARGE_RADIUS = 144;
const OIL_FIELD_EPSILON = 1e-6;

export interface OilFieldLayout {
  id: number;
  center: TileRef;
  tiles: TileRef[];
  maxReserve: number;
}

export interface OilFieldView extends OilFieldLayout {
  remainingReserve: number;
}

export interface OilFieldStateSnapshot {
  fieldId: number;
  remainingReserve: number;
}

export interface OilFieldSource {
  generate(gameMap: GameMap, config: Config): OilFieldLayout[];
}

type OilFieldTier = {
  reserve: number;
  radius: number;
};

export class ProceduralOilFieldSource implements OilFieldSource {
  generate(gameMap: GameMap, config: Config): OilFieldLayout[] {
    const oceanTileCount = countOceanTiles(gameMap);
    if (oceanTileCount === 0) {
      return [];
    }

    const desiredCount = clamp(
      Math.floor(oceanTileCount / 18_000),
      OIL_FIELD_MIN_COUNT,
      OIL_FIELD_MAX_COUNT,
    );
    const seed = simpleHash(
      [
        config.gameConfig().gameMap,
        config.gameConfig().gameMapSize,
        gameMap.width(),
        gameMap.height(),
        gameMap.numLandTiles(),
        oceanTileCount,
      ].join(":"),
    );
    const random = new PseudoRandom(seed);
    const candidates = this.buildCandidates(gameMap, random, oceanTileCount);
    if (candidates.length === 0) {
      return [];
    }

    const chosen: Array<{ tile: TileRef; tier: OilFieldTier }> = [];
    const minCenterDistance = Math.max(
      6,
      Math.floor(
        Math.sqrt(oceanTileCount / Math.max(1, desiredCount)) / 12,
      ),
    );
    const minCenterDistanceSquared = minCenterDistance ** 2;

    for (const candidate of candidates) {
      const tooClose = chosen.some(
        (field) =>
          gameMap.euclideanDistSquared(field.tile, candidate.tile) <
          minCenterDistanceSquared,
      );
      if (tooClose) {
        continue;
      }
      chosen.push({
        tile: candidate.tile,
        tier: pickTier(random),
      });
      if (chosen.length >= desiredCount) {
        break;
      }
    }

    if (chosen.length === 0) {
      chosen.push({
        tile: candidates[0]!.tile,
        tier: pickTier(random),
      });
    }

    return chosen
      .map((field, index) => {
        const tiles = Array.from(
          gameMap.circleSearch(field.tile, field.tier.radius, (tile) => {
            return gameMap.isOcean(tile);
          }),
        );
        if (tiles.length === 0) {
          return null;
        }
        tiles.sort((a, b) => a - b);
        return {
          id: index + 1,
          center: field.tile,
          tiles,
          maxReserve: field.tier.reserve,
        } satisfies OilFieldLayout;
      })
      .filter((field): field is OilFieldLayout => field !== null);
  }

  private buildCandidates(
    gameMap: GameMap,
    random: PseudoRandom,
    oceanTileCount: number,
  ): Array<{ tile: TileRef; score: number }> {
    const stride = Math.max(
      1,
      Math.floor(Math.sqrt(oceanTileCount / 1_000_000)),
    );
    const candidates: Array<{ tile: TileRef; score: number }> = [];

    for (let y = 0; y < gameMap.height(); y += stride) {
      for (let x = 0; x < gameMap.width(); x += stride) {
        const tile = gameMap.ref(x, y);
        if (!gameMap.isOcean(tile)) {
          continue;
        }
        let weight = 1.5;
        if (
          gameMap.isShoreline(tile) ||
          gameMap.neighbors(tile).some((neighbor) => gameMap.isLand(neighbor))
        ) {
          weight += 0.75;
        }
        candidates.push({
          tile,
          score: weight * (0.5 + random.nextFloat(0, 1)),
        });
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    return candidates;
  }
}

export class OilFieldManager {
  private readonly layoutsById = new Map<number, OilFieldLayout>();
  private readonly tileToFieldIds = new Map<TileRef, number[]>();
  private readonly remainingById = new Map<number, number>();

  constructor(private readonly layouts: OilFieldLayout[]) {
    for (const layout of layouts) {
      this.layoutsById.set(layout.id, layout);
      this.remainingById.set(layout.id, layout.maxReserve);
      for (const tile of layout.tiles) {
        const fieldIds = this.tileToFieldIds.get(tile) ?? [];
        fieldIds.push(layout.id);
        this.tileToFieldIds.set(tile, fieldIds);
      }
    }
  }

  static create(
    gameMap: GameMap,
    config: Config,
    source: OilFieldSource = new ProceduralOilFieldSource(),
  ): OilFieldManager {
    return new OilFieldManager(source.generate(gameMap, config));
  }

  all(): OilFieldView[] {
    return this.layouts.map((layout) => ({
      ...layout,
      remainingReserve: this.remainingReserve(layout.id),
    }));
  }

  snapshots(): OilFieldStateSnapshot[] {
    return this.layouts.map((layout) => ({
      fieldId: layout.id,
      remainingReserve: this.remainingReserve(layout.id),
    }));
  }

  layout(fieldId: number): OilFieldLayout | null {
    return this.layoutsById.get(fieldId) ?? null;
  }

  fieldIdsAt(tile: TileRef): number[] {
    return this.tileToFieldIds.get(tile) ?? [];
  }

  fieldIdAt(tile: TileRef): number | null {
    const fieldIds = this.fieldIdsAt(tile);
    if (fieldIds.length === 0) {
      return null;
    }
    const activeFieldId = fieldIds.find((fieldId) =>
      this.hasRemainingReserve(fieldId),
    );
    return activeFieldId ?? null;
  }

  fieldAt(tile: TileRef): OilFieldView | null {
    const fieldId = this.fieldIdAt(tile);
    if (fieldId === null) {
      return null;
    }
    const layout = this.layout(fieldId);
    if (layout === null) {
      return null;
    }
    return {
      ...layout,
      remainingReserve: this.remainingReserve(fieldId),
    };
  }

  remainingReserve(fieldId: number): number {
    return this.remainingById.get(fieldId) ?? 0;
  }

  hasRemainingReserve(fieldId: number): boolean {
    return this.remainingReserve(fieldId) > OIL_FIELD_EPSILON;
  }

  hasRemainingReserveAt(tile: TileRef): boolean {
    return this.fieldIdsAt(tile).some((fieldId) =>
      this.hasRemainingReserve(fieldId),
    );
  }

  setRemainingReserve(fieldId: number, remainingReserve: number): void {
    if (!this.layoutsById.has(fieldId)) {
      return;
    }
    this.remainingById.set(fieldId, Math.max(0, remainingReserve));
  }

  extract(fieldId: number, amount: number): number {
    if (amount <= 0) {
      return 0;
    }
    if (!this.layoutsById.has(fieldId)) {
      return 0;
    }

    // Infinite-oil prototype:
    // Keep the extraction API in place so the rest of the oil system can stay unchanged,
    // but stop mutating `remainingById` so fields no longer deplete over time.
    // To restore finite fields later, uncomment the reserve math below and return the
    // clamped extracted amount instead of the requested amount.
    //
    // const available = this.remainingReserve(fieldId);
    // if (available <= OIL_FIELD_EPSILON) {
    //   this.setRemainingReserve(fieldId, 0);
    //   return 0;
    // }
    // const extracted = Math.min(available, amount);
    // this.setRemainingReserve(fieldId, available - extracted);
    // return extracted;

    return amount;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function countOceanTiles(gameMap: GameMap): number {
  let oceanTiles = 0;
  gameMap.forEachTile((tile) => {
    if (gameMap.isOcean(tile)) {
      oceanTiles++;
    }
  });
  return oceanTiles;
}

function pickTier(random: PseudoRandom): OilFieldTier {
  const roll = random.nextFloat(0, 1);
  if (roll < 0.2) {
    return {
      reserve: OIL_FIELD_LARGE_RESERVE,
      radius: OIL_FIELD_LARGE_RADIUS,
    };
  }
  if (roll < 0.6) {
    return {
      reserve: OIL_FIELD_MEDIUM_RESERVE,
      radius: OIL_FIELD_MEDIUM_RADIUS,
    };
  }
  return {
    reserve: OIL_FIELD_MIN_RESERVE,
    radius: OIL_FIELD_MIN_RADIUS,
  };
}
