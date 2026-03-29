import { ToolCallsDisclosure } from "./ToolCallsDisclosure";
import { type ToolCallItem } from "./ToolCallCard";

export function ToolCallRenderer({
  items,
  streaming = false,
}: {
  items: ToolCallItem[];
  streaming?: boolean;
}) {
  if (items.length === 0) {
    return null;
  }

  return <ToolCallsDisclosure items={items} streaming={streaming} />;
}

export type { ToolCallItem } from "./ToolCallCard";
