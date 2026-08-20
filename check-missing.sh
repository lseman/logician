#!/bin/bash
src="packages/agent-core/src"
while IFS= read -r subpath; do
  subpath=$(echo "$subpath" | sed 's/"$//')
  if [ -n "$subpath" ]; then
    file="${src}/${subpath}"
    if [ -f "$file" ]; then
      echo "OK $subpath"
    else
      # Try with index.ts
      if [ -f "${file}/index.ts" ]; then
        echo "OK (index) $subpath"
      else
        echo "MISSING $subpath"
      fi
    fi
  fi
done
