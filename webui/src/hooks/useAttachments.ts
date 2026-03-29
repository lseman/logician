import { useCallback, useMemo, useState } from "react";
import type { Attachment } from "../types";
import { langFromFilename } from "../utils/messageFormatting";

export function useAttachments({
  showToast,
}: {
  showToast: (message: string, kind?: "error" | "info") => void;
}) {
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [attachLoading, setAttachLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const textExtensions = useMemo(
    () =>
      new Set([
        "py", "js", "ts", "tsx", "jsx", "java", "cpp", "c", "h", "cc", "hh", "cs", "rs", "go",
        "rb", "php", "sh", "bash", "zsh", "fish", "sql", "md", "txt", "json", "yaml", "yml",
        "toml", "ini", "cfg", "env", "csv", "html", "htm", "css", "scss", "xml", "log", "lock",
        "dockerfile", "makefile", "gitignore", "editorconfig", "prettierrc", "eslintrc",
      ]),
    [],
  );

  const processFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setAttachLoading(true);
      const pending: Promise<void>[] = [];

      for (const file of files) {
        const id = `att-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const ext = (file.name.split(".").pop() ?? "").toLowerCase();

        if (file.type.startsWith("image/")) {
          pending.push(
            new Promise((resolve) => {
              const reader = new FileReader();
              reader.onload = () => {
                setAttachments((current) => [
                  ...current,
                  {
                    id,
                    name: file.name,
                    kind: "image",
                    content: String(reader.result),
                    language: "",
                    size: file.size,
                  },
                ]);
                resolve();
              };
              reader.readAsDataURL(file);
            }),
          );
        } else if (ext === "pdf") {
          pending.push(
            (async () => {
              try {
                const form = new FormData();
                form.append("file", file);
                const res = await fetch("/api/upload", { method: "POST", body: form });
                const data = (await res.json()) as { text: string };
                setAttachments((current) => [
                  ...current,
                  {
                    id,
                    name: file.name,
                    kind: "pdf",
                    content: data.text,
                    language: "",
                    size: file.size,
                  },
                ]);
              } catch {
                showToast(`Failed to read ${file.name}`, "error");
              }
            })(),
          );
        } else if (textExtensions.has(ext) || file.type.startsWith("text/")) {
          pending.push(
            new Promise((resolve) => {
              const reader = new FileReader();
              reader.onload = () => {
                setAttachments((current) => [
                  ...current,
                  {
                    id,
                    name: file.name,
                    kind: "text",
                    content: String(reader.result),
                    language: langFromFilename(file.name),
                    size: file.size,
                  },
                ]);
                resolve();
              };
              reader.readAsText(file);
            }),
          );
        } else {
          showToast(`Unsupported file type: ${file.name}`, "error");
        }
      }

      await Promise.all(pending);
      setAttachLoading(false);
    },
    [showToast, textExtensions],
  );

  const handleFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      event.target.value = "";
      await processFiles(files);
    },
    [processFiles],
  );

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node)) {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setIsDragging(false);
      void processFiles(Array.from(event.dataTransfer.files));
    },
    [processFiles],
  );

  const clearAttachments = useCallback(() => {
    setAttachments([]);
  }, []);

  return {
    attachments,
    setAttachments,
    attachLoading,
    isDragging,
    clearAttachments,
    handleFileSelect,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  };
}
