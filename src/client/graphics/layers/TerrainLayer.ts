import { Theme } from "../../../core/configuration/Config";
import { EventBus } from "../../../core/EventBus";
import { UnitType } from "../../../core/game/Game";
import { GameView } from "../../../core/game/GameView";
import { GameUpdateType } from "../../../core/game/GameUpdates";
import { AlternateViewEvent } from "../../InputHandler";
import { TransformHandler } from "../TransformHandler";
import { UIState } from "../UIState";
import { Layer } from "./Layer";

export class TerrainLayer implements Layer {
  private canvas: HTMLCanvasElement;
  private context: CanvasRenderingContext2D;
  private imageData: ImageData;
  private oilCanvas: HTMLCanvasElement;
  private oilContext: CanvasRenderingContext2D;
  private alternativeView = false;
  private lastOilReserves = new Map<number, number>();
  private theme: Theme;

  constructor(
    private game: GameView,
    private eventBus: EventBus,
    private transformHandler: TransformHandler,
    private uiState: UIState,
  ) {}
  shouldTransform(): boolean {
    return true;
  }
  tick() {
    if (this.game.config().theme() !== this.theme) {
      this.redraw();
      return;
    }

    const oilUpdates =
      this.game.updatesSinceLastTick()?.[GameUpdateType.OilFieldState] ?? [];
    let oilChanged = false;
    for (const update of oilUpdates) {
      if (this.lastOilReserves.get(update.fieldId) !== update.remainingReserve) {
        this.lastOilReserves.set(update.fieldId, update.remainingReserve);
        oilChanged = true;
      }
    }

    if (oilChanged) {
      this.redrawOilOverlay();
    }
  }

  init() {
    this.eventBus.on(AlternateViewEvent, (event) => {
      this.alternativeView = event.alternateView;
    });
    this.redraw();
  }

  redraw(): void {
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.game.width();
    this.canvas.height = this.game.height();

    const context = this.canvas.getContext("2d", { alpha: false });
    if (context === null) throw new Error("2d context not supported");
    this.context = context;

    this.imageData = this.context.createImageData(
      this.canvas.width,
      this.canvas.height,
    );

    this.initImageData();
    this.context.putImageData(this.imageData, 0, 0);

    this.oilCanvas = document.createElement("canvas");
    this.oilCanvas.width = this.game.width();
    this.oilCanvas.height = this.game.height();

    const oilContext = this.oilCanvas.getContext("2d", { alpha: true });
    if (oilContext === null) throw new Error("2d context not supported");
    this.oilContext = oilContext;

    this.redrawOilOverlay();
  }

  initImageData() {
    this.theme = this.game.config().theme();
    this.game.forEachTile((tile) => {
      const terrainColor = this.theme.terrainColor(this.game, tile);
      // TODO: isn't tileref and index the same?
      const index = this.game.y(tile) * this.game.width() + this.game.x(tile);
      const offset = index * 4;
      this.imageData.data[offset] = terrainColor.rgba.r;
      this.imageData.data[offset + 1] = terrainColor.rgba.g;
      this.imageData.data[offset + 2] = terrainColor.rgba.b;
      this.imageData.data[offset + 3] = 255;
    });
  }

  private redrawOilOverlay() {
    const imageData = this.oilContext.createImageData(
      this.oilCanvas.width,
      this.oilCanvas.height,
    );

    for (const field of this.game.oilFields()) {
      const reserveRatio =
        field.maxReserve <= 0
          ? 0
          : clamp(field.remainingReserve / field.maxReserve, 0, 1);
      this.lastOilReserves.set(field.id, field.remainingReserve);

      if (reserveRatio <= 0) {
        continue;
      }

      const red = Math.round(70 + reserveRatio * 150);
      const green = Math.round(58 + reserveRatio * 112);
      const blue = Math.round(45 + reserveRatio * 25);
      const alpha = Math.round(30 + reserveRatio * 170);

      for (const tile of field.tiles) {
        const offset = tile * 4;
        imageData.data[offset] = red;
        imageData.data[offset + 1] = green;
        imageData.data[offset + 2] = blue;
        imageData.data[offset + 3] = alpha;
      }
    }

    this.oilContext.putImageData(imageData, 0, 0);
  }

  private shouldRenderOilOverlay(): boolean {
    return (
      this.alternativeView ||
      this.uiState.ghostStructure === UnitType.OilRig ||
      this.uiState.selectedUnitType === UnitType.OilRig
    );
  }

  renderLayer(context: CanvasRenderingContext2D) {
    if (this.transformHandler.scale < 1) {
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "low";
    } else {
      context.imageSmoothingEnabled = false;
    }
    context.drawImage(
      this.canvas,
      -this.game.width() / 2,
      -this.game.height() / 2,
      this.game.width(),
      this.game.height(),
    );
    if (this.shouldRenderOilOverlay()) {
      context.drawImage(
        this.oilCanvas,
        -this.game.width() / 2,
        -this.game.height() / 2,
        this.game.width(),
        this.game.height(),
      );
    }
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
