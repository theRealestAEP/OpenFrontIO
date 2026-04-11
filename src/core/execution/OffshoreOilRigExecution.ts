import { renderNumber } from "../../client/Utils";
import {
  Execution,
  Game,
  MessageType,
  Player,
  Unit,
  UnitType,
} from "../game/Game";
import { TileRef } from "../game/GameMap";
import { MotionPlanRecord } from "../game/MotionPlans";
import { canPlaceOilRigAt, findReachableOilRigPort } from "../game/OilRigUtils";
import { PlayerImpl } from "../game/PlayerImpl";
import { WaterPathFinder } from "../pathfinding/PathFinder";
import { PathStatus } from "../pathfinding/types";
import { OilRigExecution } from "./OilRigExecution";

export class OffshoreOilRigExecution implements Execution {
  private active = true;
  private game!: Game;
  private pathFinder!: WaterPathFinder;
  private ship: Unit | null = null;
  private rig: Unit | null = null;
  private lastMove = 0;
  private ticksUntilComplete = 0;
  private deploymentCost = 0n;
  private motionPlanId = 1;
  private motionPlanDst: TileRef | null = null;
  private readonly ticksPerMove = 1;

  constructor(
    private player: Player,
    private targetTile: TileRef,
  ) {}

  init(mg: Game, ticks: number): void {
    this.game = mg;
    this.pathFinder = new WaterPathFinder(mg);
    this.lastMove = ticks;

    const player = this.playerAsImpl();
    if (player === null) {
      this.active = false;
      return;
    }

    const port = findReachableOilRigPort(mg, player, this.targetTile);
    if (port === null) {
      this.active = false;
      return;
    }

    this.deploymentCost = mg.unitInfo(UnitType.OilRigShip).cost(mg, player);
    this.ship = player.buildUnit(UnitType.OilRigShip, port.tile(), {
      targetTile: this.targetTile,
    });
    this.ship.setTargetable(false);
    this.recordMotionPlan(ticks + this.ticksPerMove, this.ship.tile());
  }

  tick(ticks: number): void {
    if (!this.active) {
      return;
    }

    if (this.rig !== null) {
      this.tickRigConstruction();
      return;
    }

    if (this.ship === null || !this.ship.isActive()) {
      this.active = false;
      return;
    }

    if (this.pathFinder.rebuilt) {
      this.motionPlanDst = null;
    }

    if (ticks - this.lastMove < this.ticksPerMove) {
      return;
    }
    this.lastMove = ticks;

    if (!this.canContinueDeployment()) {
      this.cancelDeployment();
      return;
    }

    const result = this.pathFinder.next(this.ship.tile(), this.targetTile);
    switch (result.status) {
      case PathStatus.NEXT:
        if (this.motionPlanDst !== this.targetTile) {
          this.motionPlanId++;
          this.recordMotionPlan(ticks + this.ticksPerMove, result.node);
        }
        this.ship.move(result.node);
        return;
      case PathStatus.COMPLETE:
        if (result.node !== this.ship.tile()) {
          this.ship.move(result.node);
        }
        this.arrive();
        return;
      case PathStatus.NOT_FOUND:
        this.cancelDeployment();
        return;
    }
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }

  private canContinueDeployment(): boolean {
    return (
      this.game.isOcean(this.targetTile) &&
      canPlaceOilRigAt(this.game, this.targetTile)
    );
  }

  private arrive(): void {
    if (!this.canContinueDeployment()) {
      this.cancelDeployment();
      return;
    }

    const player = this.playerAsImpl();
    if (player === null || this.ship === null || !this.ship.isActive()) {
      this.active = false;
      return;
    }

    this.ship.delete(false);
    this.ship = null;

    this.rig = player.spawnUnitWithoutCost(
      UnitType.OilRig,
      this.targetTile,
      {},
    );
    const duration =
      this.game.unitInfo(UnitType.OilRig).constructionDuration ?? 0;
    if (duration <= 0) {
      this.game.addExecution(new OilRigExecution(this.rig));
      this.active = false;
      return;
    }

    this.rig.setUnderConstruction(true);
    this.ticksUntilComplete = duration;
  }

  private tickRigConstruction(): void {
    if (this.rig === null || !this.rig.isActive()) {
      this.active = false;
      return;
    }

    if (this.player !== this.rig.owner()) {
      this.player = this.rig.owner();
    }

    if (this.ticksUntilComplete === 0) {
      this.rig.setUnderConstruction(false);
      this.game.addExecution(new OilRigExecution(this.rig));
      this.active = false;
      return;
    }

    this.ticksUntilComplete--;
  }

  private cancelDeployment(): void {
    if (this.ship?.isActive()) {
      this.ship.delete(false);
    }
    this.ship = null;
    if (this.deploymentCost > 0n) {
      this.player.addGold(this.deploymentCost, this.targetTile);
      this.game.displayMessage(
        `Offshore oil rig deployment failed. Refunded ${renderNumber(this.deploymentCost)}.`,
        MessageType.ATTACK_FAILED,
        this.player.id(),
        this.deploymentCost,
      );
    }
    this.active = false;
  }

  private recordMotionPlan(startTick: number, from: TileRef): void {
    if (this.ship === null) {
      return;
    }

    const path = this.pathFinder.findPath(from, this.targetTile) ?? [from];
    if (path.length === 0 || path[0] !== from) {
      path.unshift(from);
    }

    const motionPlan: MotionPlanRecord = {
      kind: "grid",
      unitId: this.ship.id(),
      planId: this.motionPlanId,
      startTick,
      ticksPerStep: this.ticksPerMove,
      path,
    };
    this.game.recordMotionPlan(motionPlan);
    this.motionPlanDst = this.targetTile;
  }

  private playerAsImpl(): PlayerImpl | null {
    return this.player instanceof PlayerImpl ? this.player : null;
  }
}
