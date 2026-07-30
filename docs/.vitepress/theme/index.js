import DefaultTheme from 'vitepress/theme'
import './custom.css'
import { initMermaid } from './mermaid-init'

export default {
  ...DefaultTheme,
  setup() {
    DefaultTheme.setup?.()
    initMermaid()
  },
  onContentUpdated() {
    initMermaid()
  },
}
