import { useEffect, useState } from "react";
import mermaid from "mermaid";
import type { Artifact } from "../types";

mermaid.initialize({ startOnLoad: false, theme: "dark" });

function HtmlPreview({ content }: { content: string }) {
  return (
    <iframe
      srcDoc={content}
      sandbox="allow-scripts"
      className="artifact-iframe"
      title="HTML Preview"
    />
  );
}

function SvgPreview({ content }: { content: string }) {
  return (
    <div
      className="artifact-svg-shell"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: content }}
    />
  );
}

function MermaidPreview({ content }: { content: string }) {
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setSvg("");
    setError("");
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    mermaid.render(id, content).then(({ svg: rendered }) => {
      if (!cancelled) setSvg(rendered);
    }).catch((err: unknown) => {
      if (!cancelled) setError(err instanceof Error ? err.message : "Failed to render diagram");
    });
    return () => { cancelled = true; };
  }, [content]);

  if (error) {
    return <div className="p-4 text-sm" style={{ color: "var(--coral, #cf7668)" }}>{error}</div>;
  }
  if (!svg) {
    return (
      <div className="flex h-full items-center justify-center text-sm" style={{ color: "rgba(236,231,219,0.4)" }}>
        Rendering diagram...
      </div>
    );
  }
  return (
    <div
      className="artifact-svg-shell"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

export function ArtifactViewer({
  artifact,
  onClose,
}: {
  artifact: Artifact;
  onClose: () => void;
}) {
  const canPreview = artifact.kind === "html" || artifact.kind === "svg" || artifact.kind === "mermaid";
  const [viewMode, setViewMode] = useState<"preview" | "code">(
    canPreview ? "preview" : "code",
  );
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(artifact.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  }

  return (
    <div className="artifact-viewer">
      <div className="artifact-viewer-header">
        <div className="flex min-w-0 items-center gap-2">
          <span className="artifact-kind-badge">{artifact.kind}</span>
          <span className="artifact-title truncate">{artifact.title}</span>
        </div>
        <div className="flex items-center gap-2">
          {canPreview ? (
            <div className="artifact-mode-toggle">
              <button
                type="button"
                className={`artifact-mode-btn ${viewMode === "preview" ? "artifact-mode-btn-active" : ""}`}
                onClick={() => setViewMode("preview")}
              >
                Preview
              </button>
              <button
                type="button"
                className={`artifact-mode-btn ${viewMode === "code" ? "artifact-mode-btn-active" : ""}`}
                onClick={() => setViewMode("code")}
              >
                Code
              </button>
            </div>
          ) : null}
          <button type="button" className="artifact-action-btn" onClick={handleCopy}>
            {copied ? "Copied" : "Copy"}
          </button>
          <button type="button" className="artifact-action-btn" onClick={onClose} title="Close artifact">
            ×
          </button>
        </div>
      </div>
      <div className="artifact-viewer-body">
        {viewMode === "preview" && artifact.kind === "html" ? (
          <HtmlPreview content={artifact.content} />
        ) : viewMode === "preview" && artifact.kind === "svg" ? (
          <SvgPreview content={artifact.content} />
        ) : viewMode === "preview" && artifact.kind === "mermaid" ? (
          <MermaidPreview content={artifact.content} />
        ) : (
          <pre className="artifact-code">{artifact.content}</pre>
        )}
      </div>
    </div>
  );
}
