import {
	type Component,
	clampLineToWidth,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { type ThemeColor, theme } from "../terminal/theme.ts";

export type NotificationLevel = "info" | "success" | "warning" | "error";

export interface Notification {
	id: number;
	level: NotificationLevel;
	message: string;
}

const DISPLAY_MS = 3_500;
const MAX_VISIBLE = 3;
const MAX_HISTORY = 20;

/** Transient UI feedback. Notifications disappear from view but the last
 * MAX_HISTORY are kept in `history()` for the /notifications command. */
export class NotificationCenter implements Component {
	private notifications: Notification[] = [];
	private log: Notification[] = [];
	private nextId = 1;
	private timers = new Map<number, ReturnType<typeof setTimeout>>();
	private onInvalidate: (() => void) | null = null;
	private cachedWidth = -1;
	private cachedNotifications: Notification[] | null = null;
	private cachedLines: string[] | null = null;

	setOnInvalidate(callback: () => void): void {
		this.onInvalidate = callback;
	}

	show(
		message: string,
		level: NotificationLevel = "info",
		durationMs = DISPLAY_MS,
	): void {
		const normalized = message.trim();
		if (!normalized) return;
		const notification = { id: this.nextId++, level, message: normalized };
		this.notifications = [...this.notifications, notification].slice(
			-MAX_VISIBLE,
		);
		this.log = [...this.log, notification].slice(-MAX_HISTORY);
		const timer = setTimeout(() => this.dismiss(notification.id), durationMs);
		timer.unref?.();
		this.timers.set(notification.id, timer);
		this.onInvalidate?.();
	}

	/** Last MAX_HISTORY notifications, oldest first. */
	history(): Notification[] {
		return this.log;
	}

	dismiss(id: number): void {
		const timer = this.timers.get(id);
		if (timer) clearTimeout(timer);
		this.timers.delete(id);
		const next = this.notifications.filter(item => item.id !== id);
		if (next.length === this.notifications.length) return;
		this.notifications = next;
		this.onInvalidate?.();
	}

	clear(): void {
		for (const timer of this.timers.values()) clearTimeout(timer);
		this.timers.clear();
		this.notifications = [];
		this.onInvalidate?.();
	}

	dispose(): void {
		for (const timer of this.timers.values()) clearTimeout(timer);
		this.timers.clear();
		this.notifications = [];
		this.onInvalidate = null;
	}

	render(width: number): string[] {
		if (
			this.cachedLines !== null &&
			this.cachedWidth === width &&
			this.cachedNotifications === this.notifications
		) {
			return this.cachedLines;
		}
		this.cachedWidth = width;
		this.cachedNotifications = this.notifications;
		this.cachedLines = this.notifications.map(notification => {
			const { icon, color } = notificationStyle(notification.level);
			const content = `${theme.fg(color, icon)} ${theme.fg("text", notification.message)}${RESET}`;
			const clipped = clampLineToWidth(content, Math.max(1, width - 2));
			const line = ` ${clipped}`;
			return line + " ".repeat(Math.max(0, width - visibleWidth(line)));
		});
		return this.cachedLines;
	}
}

function notificationStyle(level: NotificationLevel): {
	icon: string;
	color: ThemeColor;
} {
	switch (level) {
		case "success":
			return { icon: "✓", color: "success" };
		case "warning":
			return { icon: "⚠", color: "warning" };
		case "error":
			return { icon: "×", color: "error" };
		default:
			return { icon: "●", color: "accent" };
	}
}
