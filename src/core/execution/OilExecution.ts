import { Execution, Game, Player, Unit, UnitType } from "../game/Game";
import {
  findReachableOilRigPort,
  isOffshoreOilRig,
  oilRigEffectiveWeight,
} from "../game/OilRigUtils";
import { toInt } from "../Util";
import { TradeShipExecution } from "./TradeShipExecution";

const OIL_TICKS_PER_SECOND = 10;
const OIL_UNITS_PER_SECOND_BASE = 30;
const OIL_GOLD_PER_UNIT = 1_200;
const OFFSHORE_CARGO_INTERVAL_TICKS = 100;

export class OilExecution implements Execution {
  private active = true;
  private game!: Game;
  private readonly offshoreCargoBuffers = new Map<Unit, number>();

  init(mg: Game): void {
    this.game = mg;
  }

  tick(ticks: number): void {
    this.pruneOffshoreCargoBuffers();

    if (ticks % OIL_TICKS_PER_SECOND === 0) {
      this.extractAndDistributeOil();
    }

    if (ticks % OFFSHORE_CARGO_INTERVAL_TICKS === 0) {
      this.launchOffshoreCargoShips();
    }
  }

  private extractAndDistributeOil(): void {
    const rigsByField = new Map<
      number,
      Map<
        Player,
        {
          effectiveRigCount: number;
          rigs: Array<{ rig: Unit; weight: number }>;
        }
      >
    >();

    for (const rig of this.game.units(UnitType.OilRig)) {
      if (!rig.isActive() || rig.isUnderConstruction()) {
        continue;
      }
      if (isOffshoreOilRig(this.game, rig) && !this.game.isOilRigActive(rig)) {
        continue;
      }
      const field = this.game.oilFieldAt(rig.tile());
      if (field === null || field.remainingReserve <= 0) {
        continue;
      }
      const perField =
        rigsByField.get(field.id) ??
        new Map<
          Player,
          {
            effectiveRigCount: number;
            rigs: Array<{ rig: Unit; weight: number }>;
          }
        >();
      const weight = oilRigEffectiveWeight(this.game, rig);
      const ownerData = perField.get(rig.owner()) ?? {
        effectiveRigCount: 0,
        rigs: [],
      };
      ownerData.effectiveRigCount += weight;
      ownerData.rigs.push({ rig, weight });
      perField.set(rig.owner(), ownerData);
      rigsByField.set(field.id, perField);
    }

    for (const [fieldId, owners] of rigsByField) {
      const field = this.game.oilFieldById(fieldId);
      if (field === null || field.remainingReserve <= 0) {
        continue;
      }

      let totalRequested = 0;
      const requestedByOwner = new Map<Player, number>();
      for (const [owner, ownerData] of owners) {
        const requested =
          OIL_UNITS_PER_SECOND_BASE * Math.log(1 + ownerData.effectiveRigCount);
        requestedByOwner.set(owner, requested);
        totalRequested += requested;
      }

      if (totalRequested <= 0) {
        continue;
      }

      const scale =
        totalRequested > field.remainingReserve
          ? field.remainingReserve / totalRequested
          : 1;

      for (const [owner, requested] of requestedByOwner) {
        const ownerData = owners.get(owner);
        if (!ownerData) {
          continue;
        }
        const extracted = this.game.extractOil(fieldId, requested * scale);
        if (extracted <= 0) {
          continue;
        }
        const gold = Math.floor(extracted * OIL_GOLD_PER_UNIT);
        if (gold <= 0) {
          continue;
        }
        this.distributeGold(ownerData.rigs, gold);
      }
    }
  }

  private distributeGold(
    rigs: Array<{ rig: Unit; weight: number }>,
    totalGold: number,
  ): void {
    const totalWeight = rigs.reduce((sum, { weight }) => sum + weight, 0);
    if (totalWeight <= 0 || rigs.length === 0) {
      return;
    }

    let remainingGold = totalGold;
    let remainingWeight = totalWeight;

    for (let i = 0; i < rigs.length; i++) {
      const { rig, weight } = rigs[i];
      const share =
        i === rigs.length - 1
          ? remainingGold
          : Math.floor((remainingGold * weight) / remainingWeight);
      remainingGold -= share;
      remainingWeight -= weight;
      if (share <= 0) {
        continue;
      }
      if (isOffshoreOilRig(this.game, rig)) {
        this.bufferOffshoreCargo(rig, share);
      } else {
        rig.owner().addGold(toInt(share), rig.tile());
        this.game.stats().goldWork(rig.owner(), share);
      }
    }
  }

  private bufferOffshoreCargo(rig: Unit, gold: number): void {
    this.offshoreCargoBuffers.set(
      rig,
      (this.offshoreCargoBuffers.get(rig) ?? 0) + gold,
    );
  }

  private launchOffshoreCargoShips(): void {
    for (const [rig, cargoGold] of Array.from(this.offshoreCargoBuffers)) {
      if (cargoGold <= 0) {
        this.offshoreCargoBuffers.delete(rig);
        continue;
      }
      if (
        !rig.isActive() ||
        rig.isUnderConstruction() ||
        !isOffshoreOilRig(this.game, rig)
      ) {
        this.offshoreCargoBuffers.delete(rig);
        continue;
      }

      const dstPort = findReachableOilRigPort(
        this.game,
        rig.owner(),
        rig.tile(),
      );
      if (dstPort === null) {
        continue;
      }

      this.offshoreCargoBuffers.delete(rig);
      this.game.addExecution(
        new TradeShipExecution(rig.owner(), rig, dstPort, {
          cargoGold,
          cargoMode: "offshore_oil",
          cargoSourceTile: rig.tile(),
          onSpawnFailed: () => {
            if (rig.isActive() && isOffshoreOilRig(this.game, rig)) {
              this.bufferOffshoreCargo(rig, cargoGold);
            }
          },
        }),
      );
    }
  }

  private pruneOffshoreCargoBuffers(): void {
    for (const rig of Array.from(this.offshoreCargoBuffers.keys())) {
      if (!rig.isActive() || rig.type() !== UnitType.OilRig) {
        this.offshoreCargoBuffers.delete(rig);
      }
    }
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }
}
