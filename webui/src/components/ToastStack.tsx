export function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: Array<{ id: number; message: string; kind: "error" | "info" }>;
  onDismiss: (id: number) => void;
}) {
  if (toasts.length === 0) {
    return null;
  }

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.kind}`}>
          <span className="flex-1">{toast.message}</span>
          <button
            type="button"
            className="ml-3 text-sand/60 transition hover:text-white"
            onClick={() => onDismiss(toast.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
