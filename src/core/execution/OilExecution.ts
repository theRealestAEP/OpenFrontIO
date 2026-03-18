import { Execution, Game, Player, Unit, UnitType } from "../game/Game";
import { toInt } from "../Util";

const OIL_TICKS_PER_SECOND = 10;
const OIL_UNITS_PER_SECOND_BASE = 30;
const OIL_GOLD_PER_UNIT = 1_200;

export class OilExecution implements Execution {
  private active = true;
  private game!: Game;

  init(mg: Game): void {
    this.game = mg;
  }

  tick(ticks: number): void {
    if (ticks % OIL_TICKS_PER_SECOND !== 0) {
      return;
    }

    const rigsByField = new Map<
      number,
      Map<Player, { effectiveRigCount: number; rigs: Unit[] }>
    >();

    for (const rig of this.game.units(UnitType.OilRig)) {
      if (!rig.isActive() || rig.isUnderConstruction()) {
        continue;
      }
      const field = this.game.oilFieldAt(rig.tile());
      if (field === null || field.remainingReserve <= 0) {
        continue;
      }
      const perField =
        rigsByField.get(field.id) ??
        new Map<Player, { effectiveRigCount: number; rigs: Unit[] }>();
      const ownerData = perField.get(rig.owner()) ?? {
        effectiveRigCount: 0,
        rigs: [],
      };
      ownerData.effectiveRigCount += rig.level();
      ownerData.rigs.push(rig);
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
        this.distributeGold(owner, ownerData.rigs, gold);
        this.game.stats().goldWork(owner, gold);
      }
    }
  }

  private distributeGold(owner: Player, rigs: Unit[], totalGold: number): void {
    const totalWeight = rigs.reduce((sum, rig) => sum + rig.level(), 0);
    if (totalWeight <= 0 || rigs.length === 0) {
      return;
    }

    let remainingGold = totalGold;
    let remainingWeight = totalWeight;

    for (let i = 0; i < rigs.length; i++) {
      const rig = rigs[i];
      const share =
        i === rigs.length - 1
          ? remainingGold
          : Math.floor((remainingGold * rig.level()) / remainingWeight);
      remainingGold -= share;
      remainingWeight -= rig.level();
      if (share > 0) {
        owner.addGold(toInt(share), rig.tile());
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
