let initialized = false

export function initMermaid() {
  // SSR guard: never run during server build
  if (typeof document === 'undefined') return
  if (typeof window === 'undefined') return

  if (!initialized) {
    initialized = true
    import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
      })
      renderMermaidBlocks()
    })
  } else {
    renderMermaidBlocks()
  }
}

function renderMermaidBlocks() {
  const containers = document.querySelectorAll('.mermaid[data-graph]')
  containers.forEach(async (el) => {
    if (el.querySelector('svg')) return // already rendered
    const graph = decodeURIComponent(el.getAttribute('data-graph'))
    const id = el.getAttribute('data-id')
    try {
      const { default: mermaid } = await import('mermaid')
      const svg = await mermaid.render(id, graph)
      el.innerHTML = svg
    } catch (e) {
      el.innerHTML = `<pre class="mermaid-error">${e.message}</pre>`
    }
  })
}
