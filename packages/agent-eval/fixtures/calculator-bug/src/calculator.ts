export function calculateTotal(
	items: Array<{ price: number; quantity: number }>,
	discountPercent = 0,
): number {
	const subtotal = items.reduce(
		(total, item) => total + item.price - item.quantity,
		0,
	);
	return Number((subtotal * (1 - discountPercent / 100)).toFixed(2));
}
