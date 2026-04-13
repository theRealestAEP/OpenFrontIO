import { WaterPathFinder } from "../pathfinding/PathFinder";
import { Game, Player, Structures, Unit, UnitType } from "./Game";
import { TileRef } from "./GameMap";
import type { OilFieldView } from "./OilField";

export const OFFSHORE_OIL_RIG_WEIGHT_MULTIPLIER = 1.5;

type PortLike = {
  tile(): TileRef;
  isActive(): boolean;
  isUnderConstruction(): boolean;
};

type PortOwnerLike<TPort extends PortLike> = {
  units(...types: UnitType[]): TPort[];
};

type WaterComponentLike = {
  getWaterComponent(tile: TileRef): number | null;
};

type OffshoreOilRigLike<TOwner> = {
  type(): UnitType;
  tile(): TileRef;
  owner(): TOwner;
  isUnderConstruction(): boolean;
};

type OffshoreOilRigGameLike = WaterComponentLike & {
  isOcean(tile: TileRef): boolean;
};

export function isOffshoreOilRig(game: Game, rig: Unit): boolean {
  return rig.type() === UnitType.OilRig && game.isOcean(rig.tile());
}

export function oilRigEffectiveWeight(game: Game, rig: Unit): number {
  const multiplier = isOffshoreOilRig(game, rig)
    ? OFFSHORE_OIL_RIG_WEIGHT_MULTIPLIER
    : 1;
  return rig.level() * multiplier;
}

export function canPlaceOilRigAt(
  game: Game,
  tile: TileRef,
  excludedShipId?: number,
): boolean {
  if (!game.isValidRef(tile)) {
    return false;
  }

  const minDistSquared = game.config().structureMinDist() ** 2;
  for (const { unit } of game.nearbyUnits(
    tile,
    game.config().structureMinDist(),
    Structures.types,
    undefined,
    true,
  )) {
    if (game.euclideanDistSquared(tile, unit.tile()) < minDistSquared) {
      return false;
    }
  }

  for (const ship of game.units(UnitType.OilRigShip)) {
    if (!ship.isActive() || ship.id() === excludedShipId) {
      continue;
    }
    const targetTile = ship.targetTile();
    if (
      targetTile !== undefined &&
      game.euclideanDistSquared(tile, targetTile) < minDistSquared
    ) {
      return false;
    }
  }

  return true;
}

export function activeOwnedPortsOnComponent(
  game: WaterComponentLike,
  player: PortOwnerLike<PortLike>,
  tile: TileRef,
): PortLike[] {
  const component = game.getWaterComponent(tile);
  if (component === null) {
    return [];
  }

  return player
    .units(UnitType.Port)
    .filter(
      (port) =>
        port.isActive() &&
        !port.isUnderConstruction() &&
        game.getWaterComponent(port.tile()) === component,
    );
}

export function hasReachableOilRigPort(
  game: WaterComponentLike,
  player: PortOwnerLike<PortLike>,
  tile: TileRef,
): boolean {
  return activeOwnedPortsOnComponent(game, player, tile).length > 0;
}

export function findReachableOilRigPort(
  game: Game,
  player: Player,
  targetTile: TileRef,
): Unit | null {
  const ports = activeOwnedPortsOnComponent(game, player, targetTile) as Unit[];
  if (ports.length === 0) {
    return null;
  }

  const pathFinder = new WaterPathFinder(game);
  let bestPort: Unit | null = null;
  let bestPathLength = Number.POSITIVE_INFINITY;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const port of ports) {
    const path = pathFinder.findPath(port.tile(), targetTile);
    if (path === null) {
      continue;
    }

    const pathLength = path.length;
    const distance = game.manhattanDist(port.tile(), targetTile);
    if (
      bestPort === null ||
      pathLength < bestPathLength ||
      (pathLength === bestPathLength && distance < bestDistance)
    ) {
      bestPort = port;
      bestPathLength = pathLength;
      bestDistance = distance;
    }
  }

  return bestPort;
}

export function isOffshoreOilRigServiced<
  TOwner extends PortOwnerLike<PortLike>,
  TRig extends OffshoreOilRigLike<TOwner>,
>(game: OffshoreOilRigGameLike, rig: TRig): boolean {
  return (
    rig.type() === UnitType.OilRig &&
    !rig.isUnderConstruction() &&
    game.isOcean(rig.tile()) &&
    hasReachableOilRigPort(game, rig.owner(), rig.tile())
  );
}

export function ownedPortWaterComponents(
  game: Game,
  player: Player,
): Set<number> {
  const components = new Set<number>();
  for (const port of player.units(UnitType.Port)) {
    if (!port.isActive() || port.isUnderConstruction()) {
      continue;
    }
    const component = game.getWaterComponent(port.tile());
    if (component !== null) {
      components.add(component);
    }
  }
  return components;
}

export function oilFieldHasReachableOffshoreTile(
  game: Game,
  field: OilFieldView,
  portComponents: Set<number>,
): boolean {
  if (portComponents.size === 0) {
    return false;
  }

  for (const tile of field.tiles) {
    if (!game.isOcean(tile)) {
      continue;
    }
    const component = game.getWaterComponent(tile);
    if (component !== null && portComponents.has(component)) {
      return true;
    }
  }

  return false;
}
