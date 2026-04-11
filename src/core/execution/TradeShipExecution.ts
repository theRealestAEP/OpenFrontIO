import { renderNumber } from "../../client/Utils";
import {
  Execution,
  Game,
  Gold,
  MessageType,
  Player,
  Unit,
  UnitType,
} from "../game/Game";
import { TileRef } from "../game/GameMap";
import { findReachableOilRigPort } from "../game/OilRigUtils";
import { WaterPathFinder } from "../pathfinding/PathFinder";
import { PathStatus } from "../pathfinding/types";
import { findClosestBy, toInt } from "../Util";

interface TradeShipExecutionOptions {
  cargoGold?: number;
  cargoMode?: "trade" | "offshore_oil";
  cargoSourceTile?: TileRef;
  onSpawnFailed?: () => void;
}

export class TradeShipExecution implements Execution {
  private active = true;
  private mg!: Game;
  private tradeShip: Unit | undefined;
  private wasCaptured = false;
  private pathFinder!: WaterPathFinder;
  private tilesTraveled = 0;
  private motionPlanId = 1;
  private motionPlanDst: TileRef | null = null;

  constructor(
    private origOwner: Player,
    private sourceStructure: Unit,
    private _dstPort: Unit,
    private options: TradeShipExecutionOptions = {},
  ) {}

  init(mg: Game, ticks: number): void {
    this.mg = mg;
    this.pathFinder = new WaterPathFinder(mg);
  }

  tick(ticks: number): void {
    if (this.pathFinder.rebuilt) {
      this.motionPlanDst = null; // Force motion plan re-recording
    }

    if (this.tradeShip === undefined) {
      const spawn = this.origOwner.canBuild(
        UnitType.TradeShip,
        this.sourceStructure.tile(),
      );
      if (spawn === false) {
        console.warn(`cannot build trade ship`);
        this.options.onSpawnFailed?.();
        this.active = false;
        return;
      }
      const tradeShipParams = {
        targetUnit: this._dstPort,
        cargoGold: this.options.cargoGold,
        cargoMode: this.cargoMode(),
        cargoSourceTile: this.options.cargoSourceTile,
        ...(!this.isOffshoreCargo() ? { lastSetSafeFromPirates: ticks } : {}),
      };
      this.tradeShip = this.origOwner.buildUnit(UnitType.TradeShip, spawn, {
        ...tradeShipParams,
      });
      if (!this.isOffshoreCargo()) {
        this.mg.stats().boatSendTrade(this.origOwner, this._dstPort.owner());
      }
    }

    if (!this.tradeShip.isActive()) {
      this.active = false;
      return;
    }

    const tradeShipOwner = this.tradeShip.owner();
    const dstPortOwner = this._dstPort.owner();
    if (this.wasCaptured !== true && this.origOwner !== tradeShipOwner) {
      // Store as variable in case ship is recaptured by previous owner
      this.wasCaptured = true;
    }

    const curTile = this.tradeShip.tile();

    if (this.isOffshoreCargo()) {
      if (!this.syncOffshoreDestination(curTile)) {
        return;
      }
    } else {
      // If a player captures another player's port while trading we should delete
      // the ship.
      if (dstPortOwner.id() === this.sourceStructure.owner().id()) {
        this.tradeShip.delete(false);
        this.active = false;
        return;
      }

      if (
        !this.wasCaptured &&
        (!this._dstPort.isActive() || !tradeShipOwner.canTrade(dstPortOwner))
      ) {
        this.tradeShip.delete(false);
        this.active = false;
        return;
      }

      if (
        this.wasCaptured &&
        (tradeShipOwner !== dstPortOwner || !this._dstPort.isActive())
      ) {
        const nearestPort = findClosestBy(
          tradeShipOwner.units(UnitType.Port),
          (port) => this.mg.manhattanDist(port.tile(), curTile),
          (port) =>
            port.isActive() &&
            !port.isMarkedForDeletion() &&
            !port.isUnderConstruction(),
        );
        if (nearestPort === null) {
          this.tradeShip.delete(false);
          this.active = false;
          return;
        } else {
          this.updateDestination(nearestPort);
        }
      }
    }

    if (curTile === this.dstPort()) {
      this.complete();
      return;
    }

    const dst = this._dstPort.tile();
    const result = this.pathFinder.next(curTile, dst);

    switch (result.status) {
      case PathStatus.NEXT:
        if (dst !== this.motionPlanDst) {
          this.motionPlanId++;
          const from = result.node;
          const path = this.pathFinder.findPath(from, dst) ?? [from];
          if (path.length === 0 || path[0] !== from) {
            path.unshift(from);
          }

          this.mg.recordMotionPlan({
            kind: "grid",
            unitId: this.tradeShip.id(),
            planId: this.motionPlanId,
            startTick: ticks + 1,
            ticksPerStep: 1,
            path,
          });
          this.motionPlanDst = dst;
        }
        // Update safeFromPirates status
        if (
          !this.isOffshoreCargo() &&
          this.mg.isWater(result.node) &&
          this.mg.isShoreline(result.node)
        ) {
          this.tradeShip.setSafeFromPirates();
        }
        this.tradeShip.move(result.node);
        this.tilesTraveled++;
        break;
      case PathStatus.COMPLETE:
        this.complete();
        return;
      case PathStatus.NOT_FOUND:
        if (
          this.isOffshoreCargo() &&
          this.rerouteOffshoreDestination(curTile)
        ) {
          const nextResult = this.pathFinder.next(
            curTile,
            this._dstPort.tile(),
          );
          if (nextResult.status !== PathStatus.NOT_FOUND) {
            return;
          }
        }
        console.warn("captured trade ship cannot find route");
        if (this.tradeShip.isActive()) {
          this.tradeShip.delete(false);
        }
        this.active = false;
        return;
    }
  }

  private complete() {
    this.active = false;
    this.tradeShip!.delete(false);
    const gold = this.completedGold();

    if (this.wasCaptured) {
      this.tradeShip!.owner().addGold(gold, this._dstPort.tile());
      this.mg.displayMessage(
        "events_display.received_gold_from_captured_ship",
        MessageType.CAPTURED_ENEMY_UNIT,
        this.tradeShip!.owner().id(),
        gold,
        {
          gold: renderNumber(gold),
          name: this.origOwner.displayName(),
        },
      );
      // Record stats
      this.mg
        .stats()
        .boatCapturedTrade(this.tradeShip!.owner(), this.origOwner, gold);
    } else if (this.isOffshoreCargo()) {
      this.tradeShip!.owner().addGold(gold, this._dstPort.tile());
      this.mg.stats().goldWork(this.tradeShip!.owner(), gold);
    } else {
      this.sourceStructure.owner().addGold(gold);
      this._dstPort.owner().addGold(gold, this._dstPort.tile());
      this.mg.displayMessage(
        "events_display.received_gold_from_trade",
        MessageType.RECEIVED_GOLD_FROM_TRADE,
        this._dstPort.owner().id(),
        gold,
        {
          gold: renderNumber(gold),
          name: this.sourceStructure.owner().displayName(),
        },
      );
      this.mg.displayMessage(
        "events_display.received_gold_from_trade",
        MessageType.RECEIVED_GOLD_FROM_TRADE,
        this.sourceStructure.owner().id(),
        gold,
        {
          gold: renderNumber(gold),
          name: this._dstPort.owner().displayName(),
        },
      );
      // Record stats
      this.mg
        .stats()
        .boatArriveTrade(
          this.sourceStructure.owner(),
          this._dstPort.owner(),
          gold,
        );
    }
    return;
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }

  dstPort(): TileRef {
    return this._dstPort.tile();
  }

  private completedGold(): Gold {
    if (this.isOffshoreCargo()) {
      return toInt(this.options.cargoGold ?? 0);
    }
    return this.mg.config().tradeShipGold(this.tilesTraveled);
  }

  private cargoMode(): "trade" | "offshore_oil" {
    return this.options.cargoMode ?? "trade";
  }

  private isOffshoreCargo(): boolean {
    return this.cargoMode() === "offshore_oil";
  }

  private syncOffshoreDestination(curTile: TileRef): boolean {
    const owner = this.tradeShip!.owner();
    if (this._dstPort.isActive() && this._dstPort.owner() === owner) {
      return true;
    }
    return this.rerouteOffshoreDestination(curTile);
  }

  private rerouteOffshoreDestination(curTile: TileRef): boolean {
    const owner = this.tradeShip!.owner();
    const nearestPort = findReachableOilRigPort(this.mg, owner, curTile);
    if (nearestPort === null) {
      this.tradeShip!.delete(false);
      this.active = false;
      return false;
    }
    this.updateDestination(nearestPort);
    return true;
  }

  private updateDestination(port: Unit): void {
    this._dstPort = port;
    this.tradeShip!.setTargetUnit(this._dstPort);
    // Plan-driven units don't emit per-tick unit updates, so force a sync for the new target.
    this.tradeShip!.touch();
    this.motionPlanDst = null;
  }
}
