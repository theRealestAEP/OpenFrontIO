import { Execution, Game, Unit, UnitType } from "../game/Game";
import { TrainStation } from "../game/TrainStation";

export class OilRigExecution implements Execution {
  private active = true;
  private game!: Game;
  private station: TrainStation | null = null;

  constructor(private rig: Unit) {}

  init(mg: Game): void {
    this.game = mg;
  }

  tick(): void {
    if (!this.rig.isActive()) {
      this.removeStation();
      this.active = false;
      return;
    }
    if (this.rig.isUnderConstruction()) {
      return;
    }

    if (this.game.isOilRigActive(this.rig)) {
      this.ensureStation();
    } else {
      this.removeStation();
    }
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }

  private ensureStation(): void {
    if (this.station !== null || this.rig.hasTrainStation()) {
      return;
    }
    const nearbyFactory = this.game.hasUnitNearby(
      this.rig.tile(),
      this.game.config().trainStationMaxRange(),
      UnitType.Factory,
    );
    if (!nearbyFactory) {
      return;
    }
    this.station = new TrainStation(this.game, this.rig);
    this.rig.setTrainStation(true);
    this.game.railNetwork().connectStation(this.station);
  }

  private removeStation(): void {
    if (!this.rig.hasTrainStation()) {
      this.station = null;
      return;
    }
    this.game.railNetwork().removeStation(this.rig);
    this.station = null;
  }
}
