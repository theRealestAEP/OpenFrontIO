import { SpatialQuery } from "../pathfinding/spatial/SpatialQuery";
import { Game, Player, TerraNullius, Unit, UnitType } from "./Game";
import { TileRef } from "./GameMap";

export interface TransportTargetInfo {
  dst: TileRef;
  offshoreRig: Unit | null;
  target: Player | TerraNullius;
}

export function canBuildTransportShip(
  game: Game,
  player: Player,
  tile: TileRef,
): TileRef | false {
  if (
    player.unitCount(UnitType.TransportShip) >= game.config().boatMaxNumber()
  ) {
    return false;
  }

  const targetInfo = resolveTransportTarget(game, player, tile);
  if (targetInfo === null) {
    return false;
  }

  const spatial = new SpatialQuery(game);
  return spatial.closestShoreByWater(player, targetInfo.dst) ?? false;
}

function landTransportTargetTile(gm: Game, tile: TileRef): TileRef | null {
  const spatial = new SpatialQuery(gm);
  return spatial.closestShore(gm.owner(tile), tile);
}

export function findCapturableOffshoreOilRig(
  game: Game,
  attacker: Player,
  tile: TileRef,
): Unit | null {
  if (!game.isOcean(tile)) {
    return null;
  }

  const rig = game
    .units(UnitType.OilRig)
    .find((unit) => unit.isActive() && unit.tile() === tile);

  if (
    rig === undefined ||
    rig.owner() === attacker ||
    rig.isUnderConstruction() ||
    !game.isOcean(rig.tile()) ||
    !attacker.canAttackPlayer(rig.owner())
  ) {
    return null;
  }

  return rig;
}

export function resolveTransportTarget(
  game: Game,
  attacker: Player,
  tile: TileRef,
): TransportTargetInfo | null {
  const offshoreRig = findCapturableOffshoreOilRig(game, attacker, tile);
  if (offshoreRig !== null) {
    return {
      dst: offshoreRig.tile(),
      offshoreRig,
      target: offshoreRig.owner(),
    };
  }

  if (game.isOcean(tile)) {
    return null;
  }

  const dst = landTransportTargetTile(game, tile);
  if (dst === null) {
    return null;
  }

  const target = game.owner(tile);
  if (target === attacker) {
    return null;
  }
  if (target.isPlayer() && !attacker.canAttackPlayer(target)) {
    return null;
  }

  return {
    dst,
    offshoreRig: null,
    target,
  };
}

export function bestShoreDeploymentSource(
  gm: Game,
  player: Player,
  dst: TileRef,
): TileRef | null {
  const spatial = new SpatialQuery(gm);
  return spatial.closestShoreByWater(player, dst);
}
